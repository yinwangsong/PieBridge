"""Twelve online epochs plus a fresh full validation pass after every epoch.

FIFO and Reorder share admission control and a continuously periodic foreground
inference stream. Decisions use live losses/EMA, rather than a prerecorded trace.
All images/proxies are resident, but validation FEATURES ARE NEVER CACHED.
Loading, initial policy inference and cost profiling precede the timed loop.
"""
import argparse
import collections
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time
import traceback

import numpy as np
import torch
import torchvision

from MNN_dataset import MNN_Dataset, MNN_Dataset_policy, Policy_Sampler
from periodic_benchmark import PeriodicInference, WindowScheduler, dump, percentiles, seed_all


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cuda_done(stream):
    event = torch.cuda.Event()
    event.record(stream)
    event.synchronize()


def cache_images(dataset, with_index=False):
    images = torch.empty((len(dataset), 3, 224, 224), device='cuda')
    labels = torch.empty(len(dataset), dtype=torch.long, device='cuda')
    for i in range(len(dataset)):
        sample = dataset[i]
        image, label = sample[1:] if with_index else sample
        images[i].copy_(image)
        labels[i] = label
    torch.cuda.synchronize()
    return images, labels


def prepare(args, output):
    started = time.perf_counter()
    seed_all(args.seed)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    backbones = []
    for i in range(10):
        model = torch.jit.load('proxynetworks/models/jit/resnet/resnet_%d.jit' % i,
                               map_location='cpu').eval().cuda()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        backbones.append(model)
    entries = torch.load('proxynetworks/pth/trainable/resnet/trainable.pt',
                         map_location='cpu')
    matched = [e for e in entries if e[0] == 'Caltech101' and e[1] == 'ours']
    assert len(matched) == 1
    _, _, lr, head = matched[0]
    head = head.eval().cuda()
    initial = copy.deepcopy(head.state_dict())
    inference_head = copy.deepcopy(head).eval()
    for parameter in inference_head.parameters():
        parameter.requires_grad_(False)
    policy = torch.load('proxynetworks/pth/policy_model/resnet/policy_resnet.pth',
                        map_location='cpu')
    dataset = MNN_Dataset_policy('datasets/Caltech101_MNN', policy, True,
                                 transform, 'resnet', 10, 'cuda:0')
    testset = MNN_Dataset('datasets/Caltech101_MNN', False, transform)
    train_images, train_labels = cache_images(dataset, with_index=True)
    eval_images, eval_labels = cache_images(testset)
    foreground_image = eval_images[:1]
    optimizer = torch.optim.SGD(head.parameters(), lr=lr)
    train_profiles = []
    for proxy, backbone in enumerate(backbones):
        head.load_state_dict(initial)
        observations = []
        for i in range(args.profile_warmup + args.profile_steps):
            before = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                features = backbone(train_images[:args.batch_size])[0]
            loss = torch.nn.functional.cross_entropy(head(features), train_labels[:args.batch_size])
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            if i >= args.profile_warmup:
                observations.append((time.perf_counter() - before) * 1000)
        train_profiles.append(dict(proxy=proxy, **percentiles(observations)))
    head.load_state_dict(initial)
    eval_observations, inference_observations = [], []
    with torch.no_grad():
        # Warm both the full evaluation batch and its non-dropped final batch.
        shapes = {args.eval_batch_size, len(testset) % args.eval_batch_size or args.eval_batch_size}
        for size in shapes:
            for _ in range(args.profile_warmup):
                head(backbones[0](eval_images[:size])[0])
        torch.cuda.synchronize()
        for i in range(args.profile_warmup + args.profile_steps):
            before = time.perf_counter()
            predictions = head(backbones[0](eval_images[:args.eval_batch_size])[0]).argmax(1)
            _ = (predictions == eval_labels[:args.eval_batch_size]).sum().item()
            torch.cuda.synchronize()
            if i >= args.profile_warmup:
                eval_observations.append((time.perf_counter() - before) * 1000)
        for i in range(args.profile_warmup + args.profile_steps):
            before = time.perf_counter()
            inference_head(backbones[0](foreground_image)[0])
            torch.cuda.synchronize()
            if i >= args.profile_warmup:
                inference_observations.append((time.perf_counter() - before) * 1000)
    head.load_state_dict(initial)
    profile = dict(
        training=train_profiles, evaluation_ms=percentiles(eval_observations),
        inference_ms=percentiles(inference_observations),
        training_images=len(dataset), validation_images=len(testset), lr=lr,
        initial_policy_histogram=dict(collections.Counter(dataset.getpolicy(i) for i in range(len(dataset)))),
        initial_policy_sha256=digest(dataset.policy),
        torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0), cudnn=torch.backends.cudnn.version(),
        nvpmodel=subprocess.run(['/usr/sbin/nvpmodel', '-q'], capture_output=True, text=True).stdout,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).with_name('periodic_benchmark.py').read_bytes()).hexdigest(),
        preparation_seconds=time.perf_counter()-started, args=vars(args),
        timing_scope='12 online epochs, 12 fresh full F0 validation passes, scheduling waits and final foreground drain; resident data/models; setup excluded')
    dump(output / 'profile.json', profile)
    print('PROFILE', json.dumps(profile), flush=True)
    return dict(backbones=backbones, head=head, initial=initial, inference_head=inference_head,
                foreground_image=foreground_image, dataset=dataset,
                train_images=train_images, train_labels=train_labels,
                eval_images=eval_images, eval_labels=eval_labels, profile=profile, lr=lr)


def await_admission(scheduler, foreground, start, period, guard, last_progress, limit):
    waited = 0.0
    while True:
        now = time.perf_counter()
        if foreground.failure:
            raise RuntimeError(foreground.failure)
        if now - start > limit:
            raise RuntimeError('Trial exceeded configured time limit')
        if now - last_progress > max(10 * period + 1, 3):
            head = scheduler.jobs[scheduler.pending[0]]
            raise RuntimeError('No progress: proxy=%s estimate_ms=%.3f period_ms=%.3f foreground_completed=%s now_s=%.3f' %
                               (head['proxy'], scheduler.estimates[head['proxy']]*1000,
                                period*1000, foreground.completed, now-start))
        due = int((now - start) // period)
        with foreground.changed:
            if foreground.completed < due:
                before = time.perf_counter()
                foreground.changed.wait(timeout=0.002)
                waited += time.perf_counter() - before
                continue
        budget = start + (due + 1) * period - now - guard
        position = scheduler.choose(budget)
        if position is not None:
            return position, waited
        before = time.perf_counter()
        time.sleep(min(0.001, max(0.0001, budget)))
        waited += time.perf_counter() - before


def status(output, **fields):
    dump(output / 'status.json', fields)


def run_trial(args, prepared, output, mode, period_ms, epochs=None,
              train_batch_limit=None, eval_limit=None, tag='main'):
    epochs = args.epochs if epochs is None else epochs
    seed_all(args.seed)
    dataset = prepared['dataset']
    dataset.policy_offset = [0] * len(dataset)
    dataset.policy_label = [0] * len(dataset)
    sampler = Policy_Sampler(dataset, args.batch_size)
    head, backbones = prepared['head'], prepared['backbones']
    initial = prepared['initial']
    head.load_state_dict(initial)
    head.eval()
    optimizer = torch.optim.SGD(head.parameters(), lr=prepared['lr'])
    stream = torch.cuda.Stream(priority=0)
    # Match the timed execution stream and indexed input path during warmup.
    # Default-stream profiling alone does not initialize every stream-local handle.
    with torch.cuda.stream(stream):
        indexes = list(range(args.batch_size))
        for backbone in backbones:
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                features = backbone(prepared['train_images'][indexes])[0]
            loss = torch.nn.functional.cross_entropy(head(features), prepared['train_labels'][indexes])
            _ = loss.item()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            for size in {args.eval_batch_size, len(prepared['eval_labels']) % args.eval_batch_size or args.eval_batch_size}:
                predicted = head(backbones[0](prepared['eval_images'][:size])[0]).argmax(1)
                _ = (predicted == prepared['eval_labels'][:size]).sum().item()
        stream.synchronize()
    head.load_state_dict(initial)
    train_costs = [p['p95'] / 1000 * args.predict_margin for p in prepared['profile']['training']]
    eval_costs = [prepared['profile']['evaluation_ms']['p95'] / 1000 * args.predict_margin]
    train_base, eval_base = list(train_costs), list(eval_costs)
    # Carry observed costs across epochs, separately for training and validation.
    train_history = {i: collections.deque(maxlen=32) for i in range(len(backbones))}
    eval_history = {0: collections.deque(maxlen=32)}
    start = time.perf_counter() + 0.1
    period, guard = period_ms / 1000, args.guard_ms / 1000
    foreground = PeriodicInference(backbones[0], prepared['inference_head'],
                                    prepared['foreground_image'], period, start,
                                    args.deadline_ms / 1000, priority=-1)
    foreground.start()
    if not foreground.ready.wait(timeout=60) or foreground.failure:
        foreground.finish(time.perf_counter())
        raise RuntimeError('Foreground thread warmup failed')
    head.load_state_dict(initial)
    torch.cuda.synchronize()
    start = time.perf_counter() + 0.05
    foreground.start_time = start
    foreground.armed.set()
    while time.perf_counter() < start:
        time.sleep(0.0002)
    ema = -1.0
    records, epochs_data = [], []
    previous_eval_end, last_progress = start, start
    stem = '%s_p%g_%s' % (tag, period_ms, mode)
    try:
        with torch.cuda.stream(stream):
            for epoch in range(1, epochs + 1):
                order = list(iter(sampler))
                batch_count = len(order) // args.batch_size
                if train_batch_limit is not None:
                    batch_count = min(batch_count, train_batch_limit)
                jobs = []
                for i in range(batch_count):
                    indexes = order[i*args.batch_size:(i+1)*args.batch_size]
                    jobs.append(dict(batch_id=i, indexes=indexes,
                                     proxy=int(dataset.policy_label[indexes[0]])))
                scheduler = WindowScheduler(jobs, train_costs, mode, args.lookahead, args.max_bypass)
                scheduler.base = train_base
                scheduler.history = train_history
                train_wait, correct = 0.0, 0
                epoch_trace, eval_trace = [], []
                # Keep dataset indexing outside candidate selection; every batch is computed once.
                while scheduler.pending:
                    pos, waited = await_admission(scheduler, foreground, start, period, guard,
                                                   last_progress, args.max_trial_seconds)
                    train_wait += waited
                    idx = scheduler.take(pos)
                    job = jobs[idx]
                    begin = time.perf_counter()
                    prediction = scheduler.estimates[job['proxy']]
                    images = prepared['train_images'][job['indexes']]
                    labels = prepared['train_labels'][job['indexes']]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        features = backbones[job['proxy']](images)[0]
                    loss = torch.nn.functional.cross_entropy(head(features), labels)
                    value = loss.item()
                    if not math.isfinite(value):
                        raise RuntimeError('Non-finite training loss')
                    backward = bool(value > 0.8 * ema)
                    if backward:
                        loss.backward()
                        optimizer.step()
                    else:
                        for image_index in job['indexes']:
                            sampler.upd_policy_with_imp(image_index, 1)
                    ema = value if ema == -1 else (2/11)*value + (9/11)*ema
                    cuda_done(stream)
                    end = time.perf_counter()
                    scheduler.observe(job['proxy'], end-begin)
                    epoch_trace.append(dict(batch_id=idx, proxy=job['proxy'], indexes=job['indexes'],
                                            backward=backward, loss=value, begin_s=begin-start,
                                            end_s=end-start, predicted_ms=prediction*1000,
                                            bypass_position=pos))
                    last_progress = end
                    if len(epoch_trace) % 100 == 0:
                        status(output, state='running', trial=stem, epoch=epoch, epochs=epochs,
                               phase='train', completed_batches=len(epoch_trace), batches=len(jobs),
                               elapsed_seconds=end-start)
                train_end = time.perf_counter()
                train_costs = list(scheduler.estimates)
                total_eval = len(prepared['eval_labels'])
                if eval_limit is not None:
                    total_eval = min(total_eval, eval_limit)
                eval_jobs = [dict(proxy=0, begin=i, end=min(i+args.eval_batch_size, total_eval))
                             for i in range(0, total_eval, args.eval_batch_size)]
                evaluation = WindowScheduler(eval_jobs, eval_costs, 'fifo', args.lookahead, args.max_bypass)
                evaluation.base = eval_base
                evaluation.history = eval_history
                eval_wait = 0.0
                # Fresh backbone forward for EVERY image in EVERY epoch. No feature cache.
                with torch.no_grad():
                    while evaluation.pending:
                        pos, waited = await_admission(evaluation, foreground, start, period, guard,
                                                       last_progress, args.max_trial_seconds)
                        eval_wait += waited
                        idx = evaluation.take(pos)
                        job = eval_jobs[idx]
                        begin = time.perf_counter()
                        prediction = evaluation.estimates[0]
                        images = prepared['eval_images'][job['begin']:job['end']]
                        labels = prepared['eval_labels'][job['begin']:job['end']]
                        predicted = head(backbones[0](images)[0]).argmax(1)
                        correct += int((predicted == labels).sum().item())
                        cuda_done(stream)
                        end = time.perf_counter()
                        evaluation.observe(0, end-begin)
                        eval_trace.append(dict(batch_id=idx, image_begin=job['begin'], image_end=job['end'],
                                               begin_s=begin-start, end_s=end-start, predicted_ms=prediction*1000))
                        last_progress = end
                        if len(eval_trace) % 100 == 0:
                            status(output, state='running', trial=stem, epoch=epoch, epochs=epochs,
                                   phase='evaluation', completed_batches=len(eval_trace), batches=len(eval_jobs),
                                   elapsed_seconds=end-start)
                eval_end = time.perf_counter()
                eval_costs = list(evaluation.estimates)
                epoch_record = dict(epoch=epoch, training_seconds=train_end-previous_eval_end,
                                    evaluation_seconds=eval_end-train_end,
                                    total_seconds=eval_end-previous_eval_end,
                                    end_s=eval_end-start, train_wait_seconds=train_wait,
                                    eval_wait_seconds=eval_wait, validation_accuracy=correct/total_eval,
                                    validation_images=total_eval, validation_batches=len(eval_jobs),
                                    training_batches=len(jobs), training_images=len(jobs)*args.batch_size,
                                    updates=sum(r['backward'] for r in epoch_trace),
                                    reordered_batches=scheduler.reordered,
                                    max_bypasses=max(scheduler.bypasses.values(), default=0),
                                    proxy_histogram=dict(collections.Counter(j['proxy'] for j in jobs)),
                                    input_plan_sha256=digest(jobs), end_policy_offsets_sha256=digest(dataset.policy_offset))
                previous_eval_end = eval_end
                records.append(dict(epoch=epoch, training_trace=epoch_trace, evaluation_trace=eval_trace))
                epochs_data.append(epoch_record)
                dump(output / (stem + '_progress.json'), dict(mode=mode, period_ms=period_ms, epochs=epochs_data))
                print('EPOCH', stem, json.dumps(epoch_record), flush=True)
        end = previous_eval_end
        foreground.finish(end)
        drained = time.perf_counter()
    except BaseException:
        error = traceback.format_exc()
        foreground.finish(time.perf_counter())
        dump(output / ('failed_' + stem + '.json'), dict(
            error=error, completed_epochs=epochs_data,
            partial_training=epoch_trace if 'epoch_trace' in locals() else [],
            partial_evaluation=eval_trace if 'eval_trace' in locals() else [],
            training_estimates=scheduler.estimates if 'scheduler' in locals() else [],
            inference_trace=foreground.records))
        raise
    inference_records = foreground.records
    responses = [r['response_ms'] for r in inference_records]
    result = dict(mode=mode, period_ms=period_ms, epochs=epochs_data,
                  training_seconds=sum(r['training_seconds'] for r in epochs_data),
                  evaluation_seconds=sum(r['evaluation_seconds'] for r in epochs_data),
                  epoch_loop_seconds=end-start, foreground_drain_seconds=drained-end,
                  total_seconds=drained-start, final_accuracy=epochs_data[-1]['validation_accuracy'],
                  inference_requests=len(inference_records), inference_response_ms=percentiles(responses),
                  deadline_ms=args.deadline_ms,
                  deadline_miss_rate=sum(r['deadline_miss'] for r in inference_records)/len(inference_records),
                  updates=sum(r['updates'] for r in epochs_data),
                  reordered_batches=sum(r['reordered_batches'] for r in epochs_data),
                  training_images=sum(r['training_images'] for r in epochs_data),
                  validation_images=sum(r['validation_images'] for r in epochs_data),
                  training_stream_priority=stream.priority, foreground_stream_priority=foreground.stream.priority,
                  live_loss_feedback=True, cached_validation_features=False, initial_seed=args.seed,
                  initial_policy_sha256=prepared['profile']['initial_policy_sha256'])
    dump(output / (stem + '.json'), dict(**result, epoch_traces=records, inference_trace=inference_records))
    print('RESULT', stem, json.dumps(result), flush=True)
    return result


def verify_trial(path, epochs, train_batches, validation_images, mode):
    d = json.loads(Path(path).read_text())
    assert len(d['epochs']) == epochs == len(d['epoch_traces'])
    assert d['cached_validation_features'] is False and d['live_loss_feedback'] is True
    assert abs(d['training_seconds'] + d['evaluation_seconds'] - d['epoch_loop_seconds']) < 1e-6
    assert abs(d['epoch_loop_seconds'] + d['foreground_drain_seconds'] - d['total_seconds']) < 1e-6
    for metrics, traces in zip(d['epochs'], d['epoch_traces']):
        training, evaluation = traces['training_trace'], traces['evaluation_trace']
        assert len(training) == train_batches
        assert sorted(r['batch_id'] for r in training) == list(range(train_batches))
        ids = [i for r in training for i in r['indexes']]
        assert len(set(ids)) == len(ids)
        assert sum(r['backward'] for r in training) == metrics['updates']
        if mode == 'fifo':
            assert [r['batch_id'] for r in training] == list(range(train_batches))
        for proxy in range(10):
            order = [r['batch_id'] for r in training if r['proxy'] == proxy]
            assert order == sorted(order)
        covered = [i for r in evaluation for i in range(r['image_begin'], r['image_end'])]
        assert covered == list(range(validation_images))
        assert metrics['validation_images'] == validation_images
        assert max(r['end_s'] for r in training) <= min(r['begin_s'] for r in evaluation)
    fg = d['inference_trace']
    assert [r['index'] for r in fg] == list(range(len(fg)))
    assert all(abs(r['release_s'] - r['index'] * d['period_ms']/1000) < 1e-5 for r in fg)
    assert fg[-1]['release_s'] <= d['epoch_loop_seconds'] + 1e-6
    assert fg[-1]['release_s'] + d['period_ms']/1000 > d['epoch_loop_seconds'] - 1e-6
    assert d['foreground_stream_priority'] == -1 and d['training_stream_priority'] == 0
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--periods-ms', type=float, nargs='+', default=[105, 135, 180])
    p.add_argument('--modes', nargs='+', choices=['fifo', 'reorder'], default=['fifo', 'reorder'])
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--eval-batch-size', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--deadline-ms', type=float, default=33.19751079194248)
    p.add_argument('--guard-ms', type=float, default=2)
    p.add_argument('--predict-margin', type=float, default=1.15)
    p.add_argument('--lookahead', type=int, default=128)
    p.add_argument('--max-bypass', type=int, default=8)
    p.add_argument('--profile-warmup', type=int, default=10)
    p.add_argument('--profile-steps', type=int, default=30)
    p.add_argument('--max-trial-seconds', type=float, default=7200)
    p.add_argument('--self-check', action='store_true')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise RuntimeError('Refusing to overwrite completed experiment results; choose a new output directory')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    status(output, state='preparing')
    try:
        prepared = prepare(args, output)
        if args.self_check or args.check_only:
            for mode in ['fifo', 'reorder']:
                run_trial(args, prepared, output, mode, 135, epochs=1,
                          train_batch_limit=8, eval_limit=17, tag='check')
                verify_trial(output / ('check_p135_%s.json' % mode), 1, 8, 17, mode)
            (output / 'self_check.txt').write_text('PASS: two real-GPU smoke trials; unique training batches; fresh validation including partial final batch; accounting and foreground releases verified.\n')
        if args.check_only:
            status(output, state='check_complete')
            return
        conditions = [(p, m) for p in args.periods_ms for m in args.modes]
        random.Random(args.seed).shuffle(conditions)
        dump(output / 'plan.json', dict(epochs=args.epochs, periods_ms=args.periods_ms,
                                        modes=args.modes, repeats=1, condition_order=conditions))
        results = []
        for period, mode in conditions:
            print('RUN', period, mode, flush=True)
            result = run_trial(args, prepared, output, mode, period)
            verify_trial(output / ('main_p%g_%s.json' % (period, mode)), args.epochs,
                         len(prepared['dataset'])//args.batch_size,
                         len(prepared['eval_labels']), mode)
            results.append(result)
            dump(output / 'summary.json', results)
        status(output, state='complete', trials=len(results))
        (output / 'verification.txt').write_text('PASS: %d trials; all epochs and full validation passes verified; elapsed time accounting, batch order and inference releases verified.\n' % len(results))
        print('FULL_BENCHMARK_COMPLETE', flush=True)
    except BaseException:
        status(output, state='failed', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
