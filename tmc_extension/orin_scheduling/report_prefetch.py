"""Verify saved epoch-local prefetch measurements without importing CUDA/Torch."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def extract_function(path, name, namespace):
    parsed = ast.parse(path.read_text())
    function = next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    p.add_argument('--smoke',action='store_true')
    args = p.parse_args()
    root = args.directory
    for name,expected in json.loads((root/'source_hashes.json').read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected, name
    namespace = {'json':json,'Path':Path}
    base = extract_function(root/'full_training_benchmark.py','verify_trial',namespace)
    namespace['full'] = SimpleNamespace(verify_trial=base)
    extended = extract_function(root/'prefetch_training_benchmark.py','verify_prefetch',namespace)
    profile = json.loads((root/'profile.json').read_text())
    prefix = 'check_' if args.smoke else 'main_'
    paths = sorted(root.glob(prefix+'p*.json'))
    paths = [p for p in paths if not p.stem.endswith('_progress')]
    assert paths
    state = json.loads((root/'status.json').read_text())['state']
    if not args.smoke:
        assert state == 'complete', 'Full experiment is not complete'
        plan = json.loads((root/'plan.json').read_text())
        expected = {(float(period), mode) for period, mode in plan['conditions']}
        actual = {(float(d['period_ms']), d['mode'])
                  for d in (json.loads(p.read_text()) for p in paths)}
        assert actual == expected and len(paths) == len(expected)
    feature_profile = {r['proxy']:r for r in json.loads((root/'feature_profile.json').read_text())}
    rows,epoch_rows,first_plans,miss_details = [],[],{},[]
    for path in paths:
        d = json.loads(path.read_text())
        epochs = 2 if args.smoke else profile['args']['epochs']
        train_batches = 32 if args.smoke else profile['training_images']//4
        validation_images = 65 if args.smoke else profile['validation_images']
        verify = extended if d['mode'].startswith('prefetch') else base
        verify(path,epochs,train_batches,validation_images,d['mode'])
        assert d['initial_seed']==profile['args']['seed']
        assert d['initial_policy_sha256']==profile['initial_policy_sha256']
        first_plans.setdefault(d['period_ms'],set()).add(d['epochs'][0]['input_plan_sha256'])
        for inference in d['inference_trace']:
            response = (inference['end_s']-inference['release_s'])*1000
            assert abs(response-inference['response_ms'])<1e-5
            assert (response>d['deadline_ms'])==inference['deadline_miss']
            if inference['deadline_miss']:
                overlap = []
                for epoch in d['epoch_traces']:
                    for phase in ['feature_trace','training_trace','evaluation_trace']:
                        for record in epoch.get(phase,[]):
                            if record['begin_s'] <= inference['end_s'] and record['end_s'] >= inference['release_s']:
                                overlap.append(dict(epoch=epoch['epoch'],phase=phase,record=record))
                miss_details.append(dict(mode=d['mode'],period_ms=d['period_ms'],
                                         inference=inference,overlapping_background=overlap))
        feature_seconds = head_seconds = whole_batch_seconds = 0.
        feature_images = 0
        grouped = {}
        offsets = [0]*profile['training_images']
        ema = -1.
        peak_bytes = peak_batches = 0
        previous_epoch_end = 0.
        for metrics,trace in zip(d['epochs'],d['epoch_traces']):
            canonical = [{k:r[k] for k in ['batch_id','indexes','proxy']}
                         for r in sorted(trace['training_trace'],key=lambda r:r['batch_id'])]
            digest = hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()
            assert digest==metrics['input_plan_sha256'], 'Input-plan digest differs from raw batches'
            for batch in trace['training_trace']:
                value = batch['loss']
                assert (value>.8*ema)==batch['backward'], 'Update decision differs from original live EMA rule'
                if not batch['backward']:
                    for index in batch['indexes']:
                        offsets[index] += 1
                ema = value if ema == -1 else (2/11)*value+(9/11)*ema
            assert hashlib.sha256(json.dumps(offsets,sort_keys=True).encode()).hexdigest()==metrics['end_policy_offsets_sha256']
            if 'feature_trace' in trace:
                # Replay allocations and consumption independently of the on-device
                # verifier. Views retain their entire merged output allocation.
                allocations, buffered = {}, set()
                events = [(r['end_s'], 'produce', r) for r in trace['feature_trace']]
                events += [(r['end_s'], 'consume', r) for r in trace['training_trace']]
                for _, event, record in sorted(events, key=lambda e:e[0]):
                    assert record['begin_s'] >= previous_epoch_end
                    if event == 'produce':
                        ids = set(record['batch_ids'])
                        assert not buffered.intersection(ids)
                        allocations[record['group_id']] = [
                            len(record['batch_ids']),
                            len(record['indexes'])*feature_profile[record['proxy']]['bytes_per_image']]
                        buffered.update(ids)
                        allocated = sum(v[1] for v in allocations.values())
                        assert len(buffered) == record['buffer_batches']
                        assert allocated == record['buffer_bytes']
                        peak_batches = max(peak_batches, len(buffered))
                        peak_bytes = max(peak_bytes, allocated)
                    else:
                        buffered.remove(record['batch_id'])
                        group = record['feature_group']
                        allocations[group][0] -= 1
                        if allocations[group][0] == 0:
                            del allocations[group]
                assert not buffered and not allocations
                feature_seconds += sum(r['end_s']-r['begin_s'] for r in trace['feature_trace'])
                feature_images += sum(len(r['indexes']) for r in trace['feature_trace'])
                for record in trace['feature_trace']:
                    key = str(len(record['batch_ids']))
                    grouped[key] = grouped.get(key,0)+1
                head_seconds += sum(r['end_s']-r['begin_s'] for r in trace['training_trace'])
            else:
                whole_batch_seconds += sum(r['end_s']-r['begin_s'] for r in trace['training_trace'])
            epoch_rows.append(dict(period_ms=d['period_ms'],mode=d['mode'],epoch=metrics['epoch'],
                training_seconds=metrics['training_seconds'],evaluation_seconds=metrics['evaluation_seconds'],
                accuracy_percent=100*metrics['validation_accuracy'],updates=metrics['updates'],
                input_plan_sha256=metrics['input_plan_sha256'],end_policy_offsets_sha256=metrics['end_policy_offsets_sha256']))
            previous_epoch_end = metrics['end_s']
        if d['mode'].startswith('prefetch'):
            assert feature_images==d['training_images']==epochs*train_batches*4
            assert peak_bytes == d['peak_feature_buffer_bytes']
            assert peak_batches == d['peak_feature_buffer_batches']
        row = dict(mode=d['mode'],period_ms=d['period_ms'],epochs=epochs,
                   training_seconds=d['training_seconds'],evaluation_seconds=d['evaluation_seconds'],
                   total_seconds=d['total_seconds'],accuracy_percent=100*d['final_accuracy'],updates=d['updates'],
                   inference_requests=d['inference_requests'],inference_p99_ms=d['inference_response_ms']['p99'],
                   inference_max_ms=d['inference_response_ms']['max'],
                   deadline_misses=sum(r['deadline_miss'] for r in d['inference_trace']),
                   deadline_miss_percent=100*d['deadline_miss_rate'],
                   peak_feature_buffer_bytes=d.get('peak_feature_buffer_bytes',0),
                   peak_feature_buffer_batches=d.get('peak_feature_buffer_batches',0),
                   feature_groups=d.get('feature_groups',0),feature_group_histogram=json.dumps(grouped,sort_keys=True),
                   feature_execution_span_seconds=feature_seconds,head_execution_span_seconds=head_seconds,
                   whole_batch_execution_span_seconds=whole_batch_seconds)
        rows.append(row)
    assert all(len(plans)==1 for plans in first_plans.values())
    comparisons = []
    for row in rows:
        if row['mode']=='fifo':continue
        baseline = next(r for r in rows if r['mode']=='fifo' and r['period_ms']==row['period_ms'])
        comparisons.append(dict(period_ms=row['period_ms'],mode=row['mode'],baseline='fifo',
            training_time_saved_percent=100*(1-row['training_seconds']/baseline['training_seconds']),
            total_time_saved_percent=100*(1-row['total_seconds']/baseline['total_seconds']),
            accuracy_difference_pp=row['accuracy_percent']-baseline['accuracy_percent'],
            deadline_miss_difference_pp=row['deadline_miss_percent']-baseline['deadline_miss_percent']))
    stem = 'smoke_' if args.smoke else ''
    for filename,contents in [('metrics',rows),('epochs',epoch_rows)]:
        with (root/(stem+filename+'.csv')).open('w') as f:
            writer = csv.DictWriter(f,fieldnames=list(contents[0]))
            writer.writeheader();writer.writerows(contents)
    (root/(stem+'comparisons.json')).write_text(json.dumps(comparisons,indent=2)+'\n')
    (root/(stem+'deadline_miss_details.json')).write_text(json.dumps(miss_details,indent=2)+'\n')
    (root/(stem+'local_verification.txt')).write_text(
        'PASS: source hashes, full training/evaluation coverage, initial state, periodic request accounting, '
        'within-epoch feature production/consumption, no cross-epoch cache, FIFO classifier updates, and elapsed accounting.\n')
    if not args.smoke:
        labels = {'fifo':'FIFO，无预取', 'reorder':'Reorder，无预取',
                  'prefetch_fifo':'FIFO，轮内预取', 'prefetch_reorder':'Reorder，轮内预取'}
        lines = ['Orin ResNet50：当前 epoch 内预取与调度完整实验', '',
                 '基线为本轮实测的无预取 FIFO。每种方法执行完整 12 个 epoch，',
                 '每轮之后重新计算全部 5,647 张验证图片。所有填充和排空开销计入时间。',
                 '没有跨 epoch 特征复用，没有修改离线模型或训练。', '',
                 '方法 | 周期(ms) | 训练(s) | 验证(s) | 总时间(s) | 准确率(%) | 更新次数 | 推理p99(ms) | 超时率(%)']
        for row in sorted(rows,key=lambda r:(r['period_ms'],list(labels).index(r['mode']))):
            lines.append(f"{labels[row['mode']]} | {row['period_ms']:g} | {row['training_seconds']:.2f} | "
                         f"{row['evaluation_seconds']:.2f} | {row['total_seconds']:.2f} | "
                         f"{row['accuracy_percent']:.2f} | {row['updates']} | "
                         f"{row['inference_p99_ms']:.2f} | {row['deadline_miss_percent']:.4f}")
        lines += ['', '相对无预取 FIFO 的时间缩短率（负值表示变慢）：']
        for c in comparisons:
            lines.append(f"{c['period_ms']:g} ms，{labels[c['mode']]}：训练 {c['training_time_saved_percent']:.2f}%，"
                         f"含验证的总时间 {c['total_time_saved_percent']:.2f}%，准确率差 {c['accuracy_difference_pp']:+.3f} 个百分点。")
        lines += ['', '前台尾延迟（不删除异常请求）：']
        for row in rows:
            lines.append(f"{labels[row['mode']]}：{row['inference_requests']} 次请求中 {row['deadline_misses']} 次超时，"
                         f"最大响应时间 {row['inference_max_ms']:.2f} ms。")
        lines += ['逐个超时请求及与其重叠的后台操作保存在 deadline_miss_details.json。',
                  'p99 较小不代表最大延迟有保证；仅凭主机时间戳不能判定异常来自 GPU、主机调度还是运行时。']
        lines += ['', '预取实现与实际占用：']
        for row in rows:
            if row['mode'].startswith('prefetch'):
                lines.append(f"{labels[row['mode']]}：特征缓冲峰值 {row['peak_feature_buffer_bytes']/1024:.2f} KiB，"
                             f"最多 {row['peak_feature_buffer_batches']} 个逻辑 batch；前向分组 {row['feature_group_histogram']}。")
        lines += ['',
                  '预取将冻结前向与分类头更新分开，并将同 proxy 的 1/2/4 个逻辑 batch 合并前向；',
                  '分类头仍按原顺序以 batch=4 更新。每张图片每轮只前向一次，每个特征只消费一次。',
                  '完整收益包含前向合并和窗口利用改善，不能单独归因于预取或异步重叠。',
                  '当前仅比较有周期推理时的四组结果，尚未测量没有前台推理时的独立收益。', '',
                  '有效性边界：每个条件只有一次完整运行，不提供置信区间或统计显著性结论。',
                  '模型/输入加载、初始策略推断、成本分析和 warmup 在计时前完成，数据与模型驻留 GPU。',
                  '全部使用同一发布模型；JIT 卷积保留内部 TF32 许可，数值结果不保证逐位一致。',
                  '分组前向和原 batch 重复前向均存在微小数值差异，可能经 loss/EMA 改变更新和后续路由。',
                  '因此时长差不能解释为执行完全相同操作序列的纯调度收益。',
                  '推理目标为 33.19751079194248 ms；每个已释放请求都处理完，未丢弃超时请求。',
                  '原始逐 batch 轨迹、生产/消费记录、源代码快照、数值检查和 SHA-256 校验均保留。']
        (root/'report_zh.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(comparisons,indent=2))
    print('VERIFIED',len(rows),'conditions')


if __name__=='__main__':
    main()
