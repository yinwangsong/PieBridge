"""Real Jetson GPU benchmark for periodic inference and PieBridge batch reordering.

Uses the released ResNet proxy family, Caltech101 images, policy model, sampler,
classifier, and SGD learning rate.  A reference pass records backward decisions;
all timing trials replay the SAME batches, proxy IDs, and backward flags.  Thus
this is a controlled fixed-work scheduling experiment, not time-to-accuracy.
Images and proxies are resident to isolate scheduling from storage/loading.
"""
import argparse
import collections
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import threading
import time
import traceback

import numpy as np
import torch
import torchvision
from PIL import Image

from MNN_dataset import MNN_Dataset, MNN_Dataset_policy, Policy_Sampler


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def percentiles(values):
    return {name: float(np.percentile(values, q)) if values else None
            for name, q in [('p50', 50), ('p95', 95), ('p99', 99), ('max', 100)]}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_step(backbone, head, optimizer, images, labels, backward=True):
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        features = backbone(images)[0]
    loss = torch.nn.functional.cross_entropy(head(features), labels)
    loss_value = loss.item()
    if not math.isfinite(loss_value):
        raise RuntimeError('Non-finite training loss; refusing to report timing as a valid run')
    if backward:
        loss.backward()
        optimizer.step()
    return loss_value


class PeriodicInference(threading.Thread):
    """Fixed releases, no dropped requests; response measured from scheduled release."""
    def __init__(self, backbone, head, images, period_s, start, deadline_s, priority=-1):
        super().__init__(daemon=True)
        self.backbone, self.head, self.images = backbone, head, images
        self.period, self.start_time, self.deadline = period_s, start, deadline_s
        self.stream = torch.cuda.Stream(priority=priority)
        self.stop_time = None
        self.completed = -1
        self.records = []
        self.failure = None
        self.changed = threading.Condition()
        self.ready = threading.Event()
        self.armed = threading.Event()

    def run(self):
        try:
            torch.cuda.set_device(0)
            with torch.cuda.stream(self.stream), torch.no_grad():
                # Warm thread-local CUDA/cuDNN handles outside measured releases.
                for _ in range(5):
                    self.head(self.backbone(self.images)[0])
                self.stream.synchronize()
                self.ready.set()
                self.armed.wait()
                index = 0
                while True:
                    release = self.start_time + index * self.period
                    with self.changed:
                        while time.perf_counter() < release:
                            if self.stop_time is not None and release > self.stop_time:
                                return
                            self.changed.wait(timeout=release - time.perf_counter())
                        if self.stop_time is not None and release > self.stop_time:
                            return
                    begin = time.perf_counter()
                    event_start = torch.cuda.Event(enable_timing=True)
                    event_end = torch.cuda.Event(enable_timing=True)
                    event_start.record(self.stream)
                    self.head(self.backbone(self.images)[0])
                    event_end.record(self.stream)
                    event_end.synchronize()
                    end = time.perf_counter()
                    self.records.append(dict(index=index,
                                             release_s=release-self.start_time,
                                             begin_s=begin-self.start_time,
                                             end_s=end-self.start_time,
                                             response_ms=(end-release)*1000,
                                             service_ms=(end-begin)*1000,
                                             gpu_ms=event_start.elapsed_time(event_end),
                                             deadline_miss=end-release > self.deadline))
                    with self.changed:
                        self.completed = index
                        self.changed.notify_all()
                    index += 1
        except BaseException:
            self.failure = traceback.format_exc()
            self.ready.set()
            with self.changed:
                self.changed.notify_all()

    def finish(self, end):
        with self.changed:
            self.stop_time = end if self.armed.is_set() else self.start_time-1
            self.changed.notify_all()
        self.armed.set()
        self.join(timeout=120)
        if self.is_alive():
            raise RuntimeError('Inference backlog did not drain in 120 seconds')
        if self.failure:
            raise RuntimeError(self.failure)


class WindowScheduler:
    """Preserve FIFO when it fits; fill otherwise-wasted tails with fitting batches.

    Only queue heads for each proxy can bypass the oldest batch.  The oldest
    batch is protected after max_bypass overtakes.  No batch is dropped.
    Prediction always budgets a full backward pass, even during trace replay.
    """
    def __init__(self, jobs, estimates, mode, lookahead, max_bypass):
        self.pending = list(range(len(jobs)))
        self.jobs = jobs
        self.estimates = list(estimates)
        self.mode = mode
        self.lookahead = lookahead
        self.max_bypass = max_bypass
        self.bypasses = collections.Counter()
        self.reordered = 0
        self.history = {i: collections.deque(maxlen=32) for i in range(len(estimates))}
        self.base = list(estimates)

    def choose(self, budget_s):
        head = self.pending[0]
        if self.mode in ('naive', 'priority', 'training_only'):
            return 0
        if self.estimates[self.jobs[head]['proxy']] <= budget_s:
            return 0
        if self.mode == 'fifo' or self.bypasses[head] >= self.max_bypass:
            return None
        seen = {self.jobs[head]['proxy']}
        feasible = []
        for pos in range(1, min(len(self.pending), self.lookahead)):
            idx = self.pending[pos]
            proxy = self.jobs[idx]['proxy']
            if proxy in seen:
                continue
            seen.add(proxy)
            cost = self.estimates[proxy]
            if cost <= budget_s:
                feasible.append((cost, -pos, pos))
        if not feasible:
            return None
        # Largest fitting cost reduces the leftover tail without an exhaustive search.
        return max(feasible)[2]

    def take(self, position):
        if position:
            self.reordered += 1
            for idx in self.pending[:position]:
                self.bypasses[idx] += 1
        return self.pending.pop(position)

    def observe(self, proxy, elapsed_s):
        self.history[proxy].append(elapsed_s)
        # Never shrink below calibration of a full training step.
        self.estimates[proxy] = max(self.base[proxy],
                                    float(np.percentile(self.history[proxy], 95)) * 1.10)


def run_trial(args, mode, period_ms, repeat, jobs, tensors, backbones, head,
              initial_state, inference_head, foreground_images, profile, lr,
              eval_features, eval_labels):
    seed_all(args.seed + repeat)
    head.load_state_dict(initial_state)
    head.eval()
    optimizer = torch.optim.SGD(head.parameters(), lr=lr)
    stream = torch.cuda.Stream(priority=0)
    # Identical warmup per condition; reset the classifier after warming.
    with torch.cuda.stream(stream):
        for i in range(10):
            proxy = i % len(backbones)
            train_step(backbones[proxy], head, optimizer, *tensors[0], backward=True)
        stream.synchronize()
    head.load_state_dict(initial_state)
    torch.cuda.synchronize()
    estimates = [p['p95_ms']/1000 * args.predict_margin
                 for p in profile['training']]
    scheduler = WindowScheduler(jobs, estimates, mode, args.lookahead, args.max_bypass)
    period = period_ms / 1000
    guard = args.guard_ms / 1000
    deadline = args.deadline_ms / 1000 if args.deadline_ms else period
    start = time.perf_counter() + 0.10
    foreground = None
    if mode != 'training_only':
        foreground = PeriodicInference(backbones[0], inference_head, foreground_images,
                                       period, start, deadline, priority=0 if mode=='naive' else -1)
        foreground.start()
        if not foreground.ready.wait(timeout=60) or foreground.failure:
            foreground.finish(time.perf_counter())
            raise RuntimeError('Foreground CUDA warmup failed')
        start = time.perf_counter() + 0.05
        foreground.start_time = start
        foreground.armed.set()
    while time.perf_counter() < start:
        time.sleep(min(0.001, max(0, start-time.perf_counter())))
    records, waits, update_count = [], 0.0, 0
    last_progress = time.perf_counter()
    with torch.cuda.stream(stream):
        while scheduler.pending:
            now = time.perf_counter()
            if foreground and foreground.failure:
                raise RuntimeError(foreground.failure)
            if now - start > args.max_trial_seconds:
                if foreground:
                    foreground.finish(now)
                raise RuntimeError('Trial exceeded time limit; scheduling may be infeasible')
            budget = float('inf')
            if mode in ('fifo', 'reorder'):
                due = int((now-start)//period)
                with foreground.changed:
                    if foreground.completed < due:
                        t = time.perf_counter()
                        foreground.changed.wait(timeout=0.002)
                        waits += time.perf_counter()-t
                        continue
                next_release = start + (due+1)*period
                budget = next_release-now-guard
            position = scheduler.choose(budget)
            if position is None:
                if now-last_progress > 10*period+1:
                    foreground.finish(time.perf_counter())
                    raise RuntimeError('Oldest batch cannot fit any full window')
                t = time.perf_counter()
                time.sleep(min(0.001, max(0.0001, budget)))
                waits += time.perf_counter()-t
                continue
            idx = scheduler.take(position)
            job = jobs[idx]
            begin = time.perf_counter()
            estimate = scheduler.estimates[job['proxy']]
            loss = train_step(backbones[job['proxy']], head, optimizer,
                              *tensors[idx], backward=job['backward'])
            event = torch.cuda.Event()
            event.record(stream)
            event.synchronize()
            end = time.perf_counter()
            if mode in ('fifo', 'reorder'):
                scheduler.observe(job['proxy'], end-begin)
            update_count += int(job['backward'])
            records.append(dict(batch_id=idx, proxy=job['proxy'],
                                backward=job['backward'], begin_s=begin-start,
                                end_s=end-start, elapsed_ms=(end-begin)*1000,
                                predicted_ms=estimate*1000, loss=loss,
                                bypass_position=position))
            last_progress = end
    end = time.perf_counter()
    if foreground:
        foreground.finish(end)
    drained = time.perf_counter()
    assert sorted(x['batch_id'] for x in records) == list(range(len(jobs)))
    fg_records = foreground.records if foreground else []
    responses = [r['response_ms'] for r in fg_records]
    service = [r['service_ms'] for r in fg_records]
    with torch.no_grad():
        validation_accuracy = (head(eval_features).argmax(dim=1) == eval_labels).float().mean().item()
    result = dict(mode=mode, period_ms=period_ms, repeat=repeat,
                  training_seconds=end-start, batches=len(jobs),
                  completion_seconds=drained-start,
                  foreground_drain_seconds=drained-end,
                  images=len(jobs)*args.batch_size, updates=update_count,
                  images_per_second=len(jobs)*args.batch_size/(end-start),
                  training_busy_seconds=sum(r['elapsed_ms'] for r in records)/1000,
                  wait_seconds=waits, reordered_batches=scheduler.reordered,
                  max_bypasses=max(scheduler.bypasses.values(), default=0),
                  inference_requests=len(fg_records),
                  foreground_stream_priority=0 if mode=='naive' else -1,
                  inference_response_ms=percentiles(responses),
                  inference_service_ms=percentiles(service), deadline_ms=deadline*1000,
                  deadline_miss_rate=(sum(r['deadline_miss'] for r in fg_records)/
                                      len(fg_records) if fg_records else 0),
                  validation_accuracy=validation_accuracy,
                  validation_images=len(eval_labels),
                  training_trace=records, inference_trace=fg_records,
                  final_loss=records[-1]['loss'])
    return result


def prepare(args):
    seed_all(args.seed)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    backbones = []
    for i in range(10):
        m = torch.jit.load('proxynetworks/models/jit/resnet/resnet_%d.jit' % i,
                           map_location='cpu').eval().cuda()
        for p in m.parameters():
            p.requires_grad_(False)
        backbones.append(m)
    trainables = torch.load('proxynetworks/pth/trainable/resnet/trainable.pt',
                           map_location='cpu')
    matched = [entry for entry in trainables
               if entry[0] == 'Caltech101' and entry[1] == 'ours']
    if len(matched) != 1:
        raise ValueError('Expected one original Caltech101/ours classifier')
    _, _, lr, head = matched[0]
    head = head.eval().cuda()
    initial = copy.deepcopy(head.state_dict())
    inference_head = copy.deepcopy(head).eval()
    for p in inference_head.parameters():
        p.requires_grad_(False)
    policy = torch.load('proxynetworks/pth/policy_model/resnet/policy_resnet.pth',
                         map_location='cpu')
    dataset = MNN_Dataset_policy('datasets/Caltech101_MNN', policy, True,
                                 transform, 'resnet', 10, 'cuda:0')
    sampler = Policy_Sampler(dataset, args.batch_size)
    # Reach a declared online epoch using the published loss/offset update rule.
    # This is ordinary on-device training, not a modification to offline training.
    warmup_epochs = []
    reference_ema = -1.0
    if args.warmup_online_epochs:
        bank = torch.empty((len(dataset),3,224,224), device='cuda')
        bank_labels = torch.empty(len(dataset), dtype=torch.long, device='cuda')
        for i in range(len(dataset)):
            _, image, label = dataset[i]
            bank[i].copy_(image)
            bank_labels[i] = label
        optimizer = torch.optim.SGD(head.parameters(), lr=lr)
        for epoch in range(args.warmup_online_epochs):
            started = time.perf_counter()
            epoch_order = list(iter(sampler))
            counts, updates, losses = collections.Counter(), 0, []
            for offset in range(0,len(epoch_order)-args.batch_size+1,args.batch_size):
                indexes = epoch_order[offset:offset+args.batch_size]
                proxy = int(dataset.policy_label[indexes[0]])
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    features = backbones[proxy](bank[indexes])[0]
                loss = torch.nn.functional.cross_entropy(head(features),bank_labels[indexes])
                value = loss.item()
                if not math.isfinite(value):
                    raise RuntimeError('Non-finite online warmup loss')
                backward = value > 0.8*reference_ema
                if backward:
                    loss.backward()
                    optimizer.step()
                    updates += 1
                else:
                    for index in indexes:
                        sampler.upd_policy_with_imp(index,1)
                reference_ema = value if reference_ema == -1 else (2/11)*value+(9/11)*reference_ema
                counts[proxy] += 1
                losses.append(value)
            torch.cuda.synchronize()
            epoch_record = dict(epoch=epoch+1,proxy_histogram=dict(counts),updates=updates,
                                mean_loss=float(np.mean(losses)),seconds=time.perf_counter()-started)
            warmup_epochs.append(epoch_record)
            print('ONLINE_EPOCH',json.dumps(epoch_record),flush=True)
        del bank, bank_labels
        initial = copy.deepcopy(head.state_dict())
        inference_head.load_state_dict(initial)
    order = list(iter(sampler))
    count = min(args.batches, len(order)//args.batch_size)
    jobs, tensors = [], []
    for idx in range(count):
        indexes = order[idx*args.batch_size:(idx+1)*args.batch_size]
        examples = [dataset[i] for i in indexes]
        images = torch.stack([x[1] for x in examples]).cuda()
        labels = torch.tensor([x[2] for x in examples], device='cuda')
        # Exactly the published select_bkb behavior: route by the first sample.
        proxy = int(dataset.policy_label[indexes[0]])
        jobs.append(dict(batch_id=idx, indexes=indexes, proxy=proxy,
                         paths=[dataset.data[i].strip().split()[0] for i in indexes]))
        tensors.append((images,labels))
    testset = MNN_Dataset('datasets/Caltech101_MNN', train=False, transform=transform)
    foreground_images = torch.stack([testset[i][0]
                                    for i in range(args.inference_batch_size)]).cuda()
    eval_count = len(testset) if args.eval_samples == 0 else min(args.eval_samples,len(testset))
    eval_indexes = random.Random(args.seed).sample(range(len(testset)),eval_count)
    eval_features, eval_labels = [], []
    with torch.no_grad():
        for offset in range(0,len(eval_indexes),16):
            samples = [testset[i] for i in eval_indexes[offset:offset+16]]
            images = torch.stack([x[0] for x in samples]).cuda()
            eval_features.append(backbones[0](images)[0])
            eval_labels.extend(x[1] for x in samples)
    eval_features = torch.cat(eval_features)
    eval_labels = torch.tensor(eval_labels,device='cuda')
    del policy, dataset, sampler
    torch.cuda.synchronize()
    # Warm every network at the actual training and inference shapes.
    optimizer = torch.optim.SGD(head.parameters(), lr=lr)
    training_profiles = []
    for i, backbone in enumerate(backbones):
        head.load_state_dict(initial)
        costs = []
        for j in range(args.profile_warmup+args.profile_steps):
            begin = time.perf_counter()
            train_step(backbone, head, optimizer, *tensors[0], backward=True)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter()-begin)*1000
            if j >= args.profile_warmup:
                costs.append(elapsed)
        training_profiles.append(dict(proxy=i, parameters=sum(p.numel() for p in backbone.parameters()),
                                      p50_ms=float(np.median(costs)),
                                      p95_ms=float(np.percentile(costs,95)), costs_ms=costs))
        print('PROFILE',i,training_profiles[-1]['p50_ms'],training_profiles[-1]['p95_ms'],flush=True)
    inference_costs = []
    with torch.no_grad():
        for i in range(20+args.profile_steps):
            begin = time.perf_counter()
            inference_head(backbones[0](foreground_images)[0])
            torch.cuda.synchronize()
            if i >= 20:
                inference_costs.append((time.perf_counter()-begin)*1000)
    head.load_state_dict(initial)
    ema = reference_ema
    optimizer = torch.optim.SGD(head.parameters(), lr=lr)
    # Record the original loss-based backward decisions using the original order.
    for job,(images,labels) in zip(jobs,tensors):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            features=backbones[job['proxy']](images)[0]
        loss=torch.nn.functional.cross_entropy(head(features),labels)
        value=loss.item()
        if not math.isfinite(value):
            raise RuntimeError('Non-finite reference loss')
        job['backward']=bool(value>0.8*ema)
        job['reference_loss']=value
        if job['backward']:
            loss.backward()
            optimizer.step()
        ema=value if ema == -1 else (2/11)*value+(9/11)*ema
    torch.cuda.synchronize()
    head.load_state_dict(initial)
    metadata=dict(training=training_profiles,
                  inference_ms=percentiles(inference_costs),
                  inference_costs_ms=inference_costs,
                  trainable_parameters=sum(p.numel() for p in head.parameters()),
                  lr=lr, proxy_histogram=dict(collections.Counter(j['proxy'] for j in jobs)),
                  online_warmup=warmup_epochs, reference_start_ema=reference_ema,
                  measured_online_epoch=args.warmup_online_epochs+1,
                  backward_batches=sum(j['backward'] for j in jobs),
                  torch=torch.__version__, torchvision=torchvision.__version__,
                  cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
                  args=vars(args), resident_proxies=10,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  experiment='fixed-work real-GPU replay; not convergence/time-to-accuracy',
                  validation_indices=eval_indexes,
                  initial_validation_accuracy=float((head(eval_features).argmax(1)==eval_labels).float().mean().item()),
                  trace_sha256=hashlib.sha256(json.dumps(jobs,sort_keys=True).encode()).hexdigest())
    return jobs,tensors,backbones,head,initial,inference_head,foreground_images,metadata,lr,eval_features,eval_labels


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--batches',type=int,default=256)
    parser.add_argument('--eval-samples',type=int,default=512,
                        help='Held-out samples to evaluate; 0 evaluates the entire test split')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--inference-batch-size',type=int,default=1)
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--warmup-online-epochs',type=int,default=0,
                        help='Run unchanged original online updates before recording the measured epoch')
    parser.add_argument('--periods-ms',type=float,nargs='*')
    parser.add_argument('--modes',nargs='+',default=['naive','priority','fifo','reorder'])
    parser.add_argument('--deadline-ms',type=float,default=0)
    parser.add_argument('--guard-ms',type=float,default=2)
    parser.add_argument('--predict-margin',type=float,default=1.15)
    parser.add_argument('--lookahead',type=int,default=128)
    parser.add_argument('--max-bypass',type=int,default=8)
    parser.add_argument('--profile-warmup',type=int,default=10)
    parser.add_argument('--profile-steps',type=int,default=30)
    parser.add_argument('--max-trial-seconds',type=float,default=300)
    parser.add_argument('--profile-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    prepared=prepare(args)
    jobs,tensors,backbones,head,initial,infhead,fgimages,profile,lr,eval_features,eval_labels=prepared
    if not args.deadline_ms:
        # A declared latency SLO independent of period; also report full response traces.
        args.deadline_ms=max(profile['inference_ms']['p99']*1.5,
                             profile['inference_ms']['p99']+2)
    profile['chosen_deadline_ms']=args.deadline_ms
    dump(output/'profile.json',profile)
    dump(output/'workload.json',jobs)
    print('WORKLOAD',profile['proxy_histogram'],'INFERENCE',profile['inference_ms'],flush=True)
    if args.profile_only:
        return
    if args.periods_ms:
        periods=args.periods_ms
    else:
        # Predeclared sweep based only on standalone profiling, not measured speedup.
        c=max(p['p95_ms'] for p in profile['training'])*args.predict_margin
        f=profile['inference_ms']['p95']
        periods=sorted(set(math.ceil((f+c*scale+args.guard_ms)/5)*5
                           for scale in [1.25,1.75,2.5]))
    dump(output/'plan.json',dict(periods_ms=periods,repeats=args.repeats,modes=args.modes))
    results=[]
    for repeat in range(args.repeats):
        conditions=[(p,m) for p in periods for m in args.modes]
        random.Random(args.seed+repeat).shuffle(conditions)
        for period,mode in conditions:
            print('RUN',repeat,period,mode,flush=True)
            result=run_trial(args,mode,period,repeat,*prepared[:7],profile,lr,eval_features,eval_labels)
            stem='r%d_p%g_%s' % (repeat,period,mode)
            dump(output/(stem+'.json'),result)
            compact={k:v for k,v in result.items()
                     if k not in ['training_trace','inference_trace']}
            results.append(compact)
            dump(output/'summary.json',results)
            print('RESULT',json.dumps(compact),flush=True)
    print('BENCHMARK_COMPLETE',flush=True)


if __name__=='__main__':
    main()
