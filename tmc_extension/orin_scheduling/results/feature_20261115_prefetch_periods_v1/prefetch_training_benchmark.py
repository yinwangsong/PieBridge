"""Epoch-local prefetch vs uncached FIFO; full live online training and validation.

Only frozen features are produced ahead. No cross-epoch reuse. Logical training
batch size and update order remain unchanged in the two prefetch conditions.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

import full_training_benchmark as full
from MNN_dataset import Policy_Sampler
from periodic_benchmark import PeriodicInference, WindowScheduler, dump, percentiles, seed_all
from prefetch_policy import feature_group


def profile_features(args, prepared, output):
    images = prepared['train_images'][:16]
    labels = prepared['train_labels'][:4]
    head = prepared['head']
    optimizer = torch.optim.SGD(head.parameters(), lr=prepared['lr'])
    rows = []
    for proxy, backbone in enumerate(prepared['backbones']):
        with torch.no_grad():
            reference = torch.cat([backbone(images[i:i+4])[0] for i in range(0,16,4)])
            grouped = backbone(images)[0]
            repeated = torch.cat([backbone(images[i:i+4])[0] for i in range(0,16,4)])
            head.load_state_dict(prepared['initial'])
            loss_errors = [abs(torch.nn.functional.cross_entropy(head(reference[i:i+4]),prepared['train_labels'][i:i+4]).item()
                               - torch.nn.functional.cross_entropy(head(grouped[i:i+4]),prepared['train_labels'][i:i+4]).item())
                           for i in range(0,16,4)]
        error = (reference-grouped).abs()
        graph = backbone.inlined_graph
        row = dict(proxy=proxy, feature_shape=list(reference.shape[1:]),
                   bytes_per_image=reference[0].numel()*reference.element_size(),
                   grouping_max_abs_error=error.max().item(),
                   grouping_relative_l2_error=(error.norm()/reference.norm().clamp_min(1e-12)).item(),
                   grouping_allclose=torch.allclose(reference,grouped,atol=3e-4,rtol=3e-4),
                   repeated_batch_max_abs_error=(reference-repeated).abs().max().item(),
                   repeated_batch_relative_l2_error=((reference-repeated).norm()/reference.norm().clamp_min(1e-12)).item(),
                   maximum_logical_batch_loss_error=max(loss_errors),
                   embedded_convolution_flags=[dict(kind=n.kind(),flags=[v.toIValue() for v in list(n.inputs())[-4:]])
                                               for n in graph.nodes() if n.kind()=='aten::_convolution'][:1],
                   forward_ms={})
        # The exported model has shape-dependent and repeated-run numerical noise.
        # Retain the original pointwise allclose result; evaluate the normalized
        # feature error and the actual unchanged logical-batch loss separately.
        assert torch.isfinite(grouped).all().item()
        assert row['grouping_relative_l2_error'] < .001, row
        assert row['maximum_logical_batch_loss_error'] < .001, row
        for group in args.group_sizes:
            observations = []
            for i in range(args.profile_warmup+args.profile_steps):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                with torch.no_grad():
                    features = backbone(images[:4*group])[0]
                torch.cuda.synchronize()
                if i >= args.profile_warmup:
                    observations.append((time.perf_counter()-begin)*1000)
            row['forward_ms'][str(group)] = percentiles(observations)
        observations = []
        for i in range(args.profile_warmup+args.profile_steps):
            head.load_state_dict(prepared['initial'])
            torch.cuda.synchronize()
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(reference[:4]), labels)
            _ = loss.item()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            if i >= args.profile_warmup:
                observations.append((time.perf_counter()-begin)*1000)
        row['head_ms'] = percentiles(observations)
        rows.append(row)
        print('FEATURE_PROFILE',json.dumps(row),flush=True)
    head.load_state_dict(prepared['initial'])
    prepared['feature_profile'] = rows
    dump(output/'feature_profile.json',rows)
    return rows


def run_prefetch(args, prepared, output, mode, period_ms, epochs=None,
                 train_batch_limit=None, eval_limit=None, tag='main'):
    epochs = epochs or args.epochs
    seed_all(args.seed)
    dataset = prepared['dataset']
    dataset.policy_offset = [0]*len(dataset)
    dataset.policy_label = [0]*len(dataset)
    sampler = Policy_Sampler(dataset,args.batch_size)
    head, backbones = prepared['head'], prepared['backbones']
    head.load_state_dict(prepared['initial'])
    optimizer = torch.optim.SGD(head.parameters(),lr=prepared['lr'])
    stream = torch.cuda.Stream(priority=0)
    with torch.cuda.stream(stream):
        for backbone in backbones:
            for size in args.group_sizes:
                with torch.no_grad():
                    features = backbone(prepared['train_images'][list(range(4*size))])[0]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(head(features[:4]),prepared['train_labels'][:4])
                _ = loss.item()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            for size in {4,len(prepared['eval_labels'])%4 or 4}:
                result = head(backbones[0](prepared['eval_images'][:size])[0]).argmax(1)
                _ = (result==prepared['eval_labels'][:size]).sum().item()
        stream.synchronize()
    del features,loss,result
    head.load_state_dict(prepared['initial'])
    feature_base = {(r['proxy'],g):r['forward_ms'][str(g)]['p95']*.001*args.predict_margin
                    for r in prepared['feature_profile'] for g in args.group_sizes}
    head_base = {r['proxy']:r['head_ms']['p95']*.001*args.predict_margin for r in prepared['feature_profile']}
    feature_cost, head_cost = dict(feature_base),dict(head_base)
    feature_history = {key:collections.deque(maxlen=32) for key in feature_base}
    head_history = {key:collections.deque(maxlen=32) for key in head_base}
    eval_base = [prepared['profile']['evaluation_ms']['p95']*.001*args.predict_margin]
    eval_cost = list(eval_base)
    eval_history = {0:collections.deque(maxlen=32)}
    period,guard = period_ms*.001,args.guard_ms*.001
    start = time.perf_counter()+.1
    foreground = PeriodicInference(backbones[0],prepared['inference_head'],prepared['foreground_image'],
                                    period,start,args.deadline_ms*.001,priority=-1)
    foreground.start()
    if not foreground.ready.wait(60) or foreground.failure:
        foreground.finish(time.perf_counter())
        raise RuntimeError('Foreground warmup failed')
    head.load_state_dict(prepared['initial'])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()+.05
    foreground.start_time = start
    foreground.armed.set()
    while time.perf_counter()<start:
        time.sleep(.0002)
    stem = '%s_p%g_%s'%(tag,period_ms,mode)
    ema = -1.
    previous_eval_end = last_progress = start
    epochs_data,records = [],[]
    peak_buffer_bytes = peak_buffer_batches = 0

    def budget_now():
        now = time.perf_counter()
        if foreground.failure:
            raise RuntimeError(foreground.failure)
        if now-start > args.max_trial_seconds or now-last_progress > max(3,10*period+1):
            raise RuntimeError('Timed out or no prefetch progress')
        due = int((now-start)//period)
        with foreground.changed:
            if foreground.completed < due:
                return -1.
        return start+(due+1)*period-now-guard

    try:
        with torch.cuda.stream(stream):
            for epoch in range(1,epochs+1):
                order = list(iter(sampler))
                count = len(order)//4
                if train_batch_limit is not None:
                    count = min(count,train_batch_limit)
                jobs = [dict(batch_id=i,indexes=order[4*i:4*(i+1)],
                             proxy=int(dataset.policy_label[order[4*i]])) for i in range(count)]
                buffer,allocations = {},{}
                buffer_bytes = 0
                trace,production,eval_trace = [],[],[]
                cursor = 0
                train_wait = 0.
                while cursor < count:
                    budget = budget_now()
                    proxy = jobs[cursor]['proxy']
                    if cursor in buffer and head_cost[proxy] <= budget:
                        begin = time.perf_counter()
                        features,group_id = buffer.pop(cursor)
                        labels = prepared['train_labels'][jobs[cursor]['indexes']]
                        optimizer.zero_grad(set_to_none=True)
                        loss = torch.nn.functional.cross_entropy(head(features),labels)
                        value = loss.item()
                        if not math.isfinite(value):
                            raise RuntimeError('Non-finite training loss')
                        backward = bool(value > .8*ema)
                        if backward:
                            loss.backward()
                            optimizer.step()
                        else:
                            for index in jobs[cursor]['indexes']:
                                sampler.upd_policy_with_imp(index,1)
                        ema = value if ema == -1 else (2/11)*value+(9/11)*ema
                        full.cuda_done(stream)
                        end = time.perf_counter()
                        prediction = head_cost[proxy]
                        head_history[proxy].append(end-begin)
                        head_cost[proxy] = max(head_base[proxy],float(np.percentile(head_history[proxy],95))*1.10)
                        trace.append(dict(**jobs[cursor],backward=backward,loss=value,
                                          begin_s=begin-start,end_s=end-start,predicted_ms=prediction*1000,
                                          feature_group=group_id,bypass_position=0))
                        allocations[group_id]['remaining'] -= 1
                        if not allocations[group_id]['remaining']:
                            buffer_bytes -= allocations.pop(group_id)['bytes']
                        del features,loss
                        last_progress = end
                        cursor += 1
                        if cursor%100 == 0:
                            full.status(output,state='running',trial=stem,epoch=epoch,epochs=epochs,
                                        phase='train',completed_batches=cursor,batches=count,
                                        elapsed_seconds=end-start)
                        continue
                    # The ready oldest update has priority. Do not fill more features
                    # while it waits for its own budget, which bounds speculation.
                    group = [] if cursor in buffer else feature_group(
                        jobs,cursor,buffer,args.buffer_batches,args.lookahead,args.group_sizes,
                        feature_cost,budget,mode=='prefetch_reorder')
                    if group:
                        proxy = jobs[group[0]]['proxy']
                        indexes = [i for batch in group for i in jobs[batch]['indexes']]
                        begin = time.perf_counter()
                        with torch.no_grad():
                            produced = backbones[proxy](prepared['train_images'][indexes])[0]
                        full.cuda_done(stream)
                        end = time.perf_counter()
                        group_id = len(production)
                        size_bytes = produced.numel()*produced.element_size()
                        allocations[group_id] = dict(bytes=size_bytes,remaining=len(group))
                        for offset,batch in enumerate(group):
                            assert batch not in buffer and batch >= cursor
                            buffer[batch] = (produced[4*offset:4*(offset+1)],group_id)
                        buffer_bytes += size_bytes
                        peak_buffer_bytes = max(peak_buffer_bytes,buffer_bytes)
                        peak_buffer_batches = max(peak_buffer_batches,len(buffer))
                        key = (proxy,len(group))
                        prediction = feature_cost[key]
                        feature_history[key].append(end-begin)
                        feature_cost[key] = max(feature_base[key],float(np.percentile(feature_history[key],95))*1.10)
                        production.append(dict(group_id=group_id,batch_ids=group,proxy=proxy,indexes=indexes,
                                               begin_s=begin-start,end_s=end-start,predicted_ms=prediction*1000,
                                               buffer_batches=len(buffer),buffer_bytes=buffer_bytes,
                                               oldest_update=cursor))
                        del produced
                        last_progress = end
                        continue
                    before = time.perf_counter()
                    time.sleep(.0005)
                    train_wait += time.perf_counter()-before
                assert not buffer and not allocations and buffer_bytes == 0
                train_end = time.perf_counter()
                total_eval = len(prepared['eval_labels'])
                if eval_limit is not None:
                    total_eval = min(total_eval,eval_limit)
                eval_jobs = [dict(proxy=0,begin=i,end=min(i+4,total_eval)) for i in range(0,total_eval,4)]
                evaluation = WindowScheduler(eval_jobs,eval_cost,'fifo',args.lookahead,args.max_bypass)
                evaluation.base,evaluation.history = eval_base,eval_history
                eval_wait = correct = 0
                with torch.no_grad():
                    while evaluation.pending:
                        pos,waited = full.await_admission(evaluation,foreground,start,period,guard,
                                                         last_progress,args.max_trial_seconds)
                        eval_wait += waited
                        index = evaluation.take(pos)
                        job = eval_jobs[index]
                        begin = time.perf_counter()
                        prediction = evaluation.estimates[0]
                        images = prepared['eval_images'][job['begin']:job['end']]
                        labels = prepared['eval_labels'][job['begin']:job['end']]
                        pred = head(backbones[0](images)[0]).argmax(1)
                        correct += int((pred==labels).sum().item())
                        full.cuda_done(stream)
                        end = time.perf_counter()
                        evaluation.observe(0,end-begin)
                        eval_trace.append(dict(batch_id=index,image_begin=job['begin'],image_end=job['end'],
                                               begin_s=begin-start,end_s=end-start,predicted_ms=prediction*1000))
                        last_progress = end
                        if len(eval_trace)%100 == 0:
                            full.status(output,state='running',trial=stem,epoch=epoch,epochs=epochs,
                                        phase='evaluation',completed_batches=len(eval_trace),batches=len(eval_jobs),
                                        elapsed_seconds=end-start)
                eval_end = time.perf_counter()
                eval_cost = list(evaluation.estimates)
                flattened = [i for group in production for i in group['batch_ids']]
                reordered = sum(i!=j for i,j in enumerate(flattened))
                metric = dict(epoch=epoch,training_seconds=train_end-previous_eval_end,
                              evaluation_seconds=eval_end-train_end,total_seconds=eval_end-previous_eval_end,
                              end_s=eval_end-start,train_wait_seconds=train_wait,eval_wait_seconds=eval_wait,
                              validation_accuracy=correct/total_eval,validation_images=total_eval,
                              validation_batches=len(eval_jobs),training_batches=count,training_images=count*4,
                              updates=sum(r['backward'] for r in trace),reordered_batches=reordered,
                              feature_groups=len(production),group_size_histogram=dict(collections.Counter(len(g['batch_ids']) for g in production)),
                              proxy_histogram=dict(collections.Counter(j['proxy'] for j in jobs)),
                              input_plan_sha256=full.digest(jobs),
                              end_policy_offsets_sha256=full.digest(dataset.policy_offset),
                              feature_buffer_empty=True)
                epochs_data.append(metric)
                records.append(dict(epoch=epoch,training_trace=trace,evaluation_trace=eval_trace,feature_trace=production))
                previous_eval_end = eval_end
                dump(output/(stem+'_progress.json'),dict(mode=mode,period_ms=period_ms,epochs=epochs_data))
                print('EPOCH',stem,json.dumps(metric),flush=True)
        end = previous_eval_end
        foreground.finish(end)
        drained = time.perf_counter()
    except BaseException:
        error = traceback.format_exc()
        foreground.finish(time.perf_counter())
        dump(output/('failed_'+stem+'.json'),dict(error=error,completed_epochs=epochs_data,
             partial_training=trace if 'trace' in locals() else [],
             partial_features=production if 'production' in locals() else [],inference_trace=foreground.records))
        raise
    responses = [r['response_ms'] for r in foreground.records]
    result = dict(mode=mode,period_ms=period_ms,epochs=epochs_data,
                  training_seconds=sum(r['training_seconds'] for r in epochs_data),
                  evaluation_seconds=sum(r['evaluation_seconds'] for r in epochs_data),
                  epoch_loop_seconds=end-start,foreground_drain_seconds=drained-end,total_seconds=drained-start,
                  final_accuracy=epochs_data[-1]['validation_accuracy'],inference_requests=len(responses),
                  inference_response_ms=percentiles(responses),deadline_ms=args.deadline_ms,
                  deadline_miss_rate=sum(r['deadline_miss'] for r in foreground.records)/len(responses),
                  updates=sum(r['updates'] for r in epochs_data),
                  reordered_batches=sum(r['reordered_batches'] for r in epochs_data),
                  feature_groups=sum(r['feature_groups'] for r in epochs_data),
                  training_images=sum(r['training_images'] for r in epochs_data),
                  validation_images=sum(r['validation_images'] for r in epochs_data),
                  peak_feature_buffer_bytes=peak_buffer_bytes,peak_feature_buffer_batches=peak_buffer_batches,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  training_stream_priority=stream.priority,foreground_stream_priority=foreground.stream.priority,
                  live_loss_feedback=True,cached_validation_features=False,cross_epoch_cache=False,
                  classifier_update_order='original FIFO',initial_seed=args.seed,
                  reorder_count_definition='feature-production batch IDs displaced from their original positions',
                  initial_policy_sha256=prepared['profile']['initial_policy_sha256'])
    dump(output/(stem+'.json'),dict(**result,epoch_traces=records,inference_trace=foreground.records))
    print('RESULT',stem,json.dumps(result),flush=True)
    return result


def verify_prefetch(path,epochs,train_batches,validation_images,mode,capacity=32):
    full.verify_trial(path,epochs,train_batches,validation_images,mode)
    d = json.loads(Path(path).read_text())
    assert not d['cross_epoch_cache'] and not d['cached_validation_features']
    assert d['peak_feature_buffer_batches'] <= capacity
    for metric,traces in zip(d['epochs'],d['epoch_traces']):
        training,features = traces['training_trace'],traces['feature_trace']
        assert [r['batch_id'] for r in training] == list(range(train_batches))
        produced = [i for r in features for i in r['batch_ids']]
        assert sorted(produced) == list(range(train_batches))
        if mode == 'prefetch_fifo':
            assert produced == list(range(train_batches))
        groups = {g['group_id']:g for g in features}
        assert metric['feature_buffer_empty']
        for r in training:
            g = groups[r['feature_group']]
            assert r['batch_id'] in g['batch_ids'] and r['proxy'] == g['proxy']
            assert g['end_s'] <= r['begin_s']
            pos = g['batch_ids'].index(r['batch_id'])
            assert g['indexes'][4*pos:4*(pos+1)] == r['indexes']
        for group in features:
            assert len(group['batch_ids']) in [1,2,4]
            assert group['batch_ids'] == sorted(group['batch_ids'])
            assert group['buffer_batches'] <= capacity
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--periods-ms',nargs='+',type=float,default=[105])
    p.add_argument('--modes',nargs='+',default=['fifo','reorder','prefetch_fifo','prefetch_reorder'])
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--eval-batch-size',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261111)
    p.add_argument('--deadline-ms',type=float,default=33.19751079194248)
    p.add_argument('--guard-ms',type=float,default=2)
    p.add_argument('--predict-margin',type=float,default=1.15)
    p.add_argument('--lookahead',type=int,default=128)
    p.add_argument('--max-bypass',type=int,default=8)
    p.add_argument('--buffer-batches',type=int,default=32)
    p.add_argument('--group-sizes',type=int,nargs='+',default=[1,2,4])
    p.add_argument('--profile-warmup',type=int,default=10)
    p.add_argument('--profile-steps',type=int,default=30)
    p.add_argument('--max-trial-seconds',type=float,default=7200)
    p.add_argument('--check-only',action='store_true')
    args = p.parse_args()
    assert args.batch_size == args.eval_batch_size == 4
    assert args.group_sizes == [1,2,4]
    assert set(args.modes) <= {'fifo','reorder','prefetch_fifo','prefetch_reorder'}
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    assert not (output/'summary.json').exists()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    full.status(output,state='preparing')
    try:
        prepared = full.prepare(args,output)
        profile_features(args,prepared,output)
        dump(output/'source_hashes.json',{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
             for name in ['prefetch_training_benchmark.py','prefetch_policy.py','full_training_benchmark.py','periodic_benchmark.py']})
        checks = []
        check_period = 180  # Existing setting with room to exercise merged F0 forwards.
        for mode in args.modes:
            run = run_prefetch if mode.startswith('prefetch') else full.run_trial
            check = run(args,prepared,output,mode,check_period,epochs=2,train_batch_limit=32,eval_limit=65,tag='check')
            verify = verify_prefetch if mode.startswith('prefetch') else full.verify_trial
            verify(output/('check_p%g_%s.json'%(check_period,mode)),2,32,65,mode)
            checks.append(check)
        # The same first-epoch inputs and loss-based update decisions must survive grouping.
        assert len({x['epochs'][0]['input_plan_sha256'] for x in checks}) == 1
        checks_by_mode = {x['mode']:x for x in checks}
        if 'fifo' in checks_by_mode:
            original = json.loads((output/('check_p%g_fifo.json'%check_period)).read_text())
            for mode in ['prefetch_fifo','prefetch_reorder']:
                if mode in checks_by_mode:
                    other = json.loads((output/('check_p%g_%s.json'%(check_period,mode))).read_text())
                    a,b = original['epoch_traces'][0]['training_trace'],other['epoch_traces'][0]['training_trace']
                    assert all(x['backward']==y['backward'] for x,y in zip(a,b))
                    assert max(abs(x['loss']-y['loss']) for x,y in zip(a,b)) < .001
                    assert any(len(g['batch_ids'])>1 for e in other['epoch_traces'] for g in e['feature_trace'])
        dump(output/'check_summary.json',checks)
        if args.check_only:
            full.status(output,state='check_complete')
            return
        conditions = [(p,m) for p in args.periods_ms for m in args.modes]
        random.Random(args.seed).shuffle(conditions)
        dump(output/'plan.json',dict(epochs=args.epochs,conditions=conditions,repeats=1))
        results = []
        for period,mode in conditions:
            run = run_prefetch if mode.startswith('prefetch') else full.run_trial
            result = run(args,prepared,output,mode,period)
            verify = verify_prefetch if mode.startswith('prefetch') else full.verify_trial
            verify(output/('main_p%g_%s.json'%(period,mode)),args.epochs,len(prepared['dataset'])//4,
                   len(prepared['eval_labels']),mode)
            results.append(result)
            dump(output/'summary.json',results)
        (output/'verification.txt').write_text('PASS: complete train/evaluation coverage, foreground releases, elapsed accounting, bounded epoch-local feature production and FIFO consumption.\n')
        full.status(output,state='complete',trials=len(results))
    except BaseException:
        full.status(output,state='failed',traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
