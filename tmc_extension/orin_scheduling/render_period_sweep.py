"""Render Table 8 only from complete, verified full-training period pairs."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'results/feature_20261113_prefetch_full_v2'
NEW = ROOT / 'results/feature_20261115_prefetch_periods_v1'
PERIODS = (105, 135, 180)
MODES = ('fifo', 'prefetch_reorder')


def read_verified(directory):
    assert json.loads((directory / 'status.json').read_text())['state'] == 'complete'
    assert (directory / 'local_verification.txt').read_text().startswith('PASS:')
    hashes = json.loads((directory / 'source_hashes.json').read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
    profile = json.loads((directory / 'profile.json').read_text())
    with (directory / 'metrics.csv').open() as stream:
        rows = {(float(r['period_ms']), r['mode']): r for r in csv.DictReader(stream)}
    return profile, hashes, rows


def main():
    old_profile, old_hashes, old_rows = read_verified(OLD)
    new_profile, new_hashes, new_rows = read_verified(NEW)
    assert old_hashes == new_hashes, 'Benchmark sources differ across campaigns'
    for key in ('training_images', 'validation_images', 'lr', 'initial_policy_sha256',
                'torch', 'torchvision', 'cuda', 'device', 'cudnn', 'git_commit'):
        assert old_profile[key] == new_profile[key], key
    for key, value in old_profile['args'].items():
        if key not in ('output', 'periods_ms', 'modes'):
            assert new_profile['args'][key] == value, key
    assert new_profile['args']['epochs'] == 12
    assert set(new_rows) == {(float(p), m) for p in (135, 180) for m in MODES}

    comparisons, table = [], []
    first_plan = None
    for period in PERIODS:
        directory, rows = (OLD, old_rows) if period == 105 else (NEW, new_rows)
        pair = {}
        for mode in MODES:
            row = rows[(float(period), mode)]
            path = directory / f'main_p{period}_{mode}.json'
            run = json.loads(path.read_text())
            assert int(row['epochs']) == len(run['epochs']) == 12
            assert run['period_ms'] == period and run['mode'] == mode
            assert float(row['training_seconds']) == run['training_seconds']
            assert run['training_images'] == 12 * 757 * 4
            assert run['validation_images'] == 12 * 5647
            digest = run['epochs'][0]['input_plan_sha256']
            first_plan = first_plan or digest
            assert digest == first_plan, 'First-epoch input plans differ'
            pair[mode] = run
        baseline, combined = pair['fifo'], pair['prefetch_reorder']
        saved = 100 * (1 - combined['training_seconds'] / baseline['training_seconds'])
        comparisons.append(dict(
            period_ms=period, fifo_training_seconds=baseline['training_seconds'],
            prefetch_reorder_training_seconds=combined['training_seconds'],
            training_time_saved_percent=saved,
            fifo_accuracy_percent=100 * baseline['final_accuracy'],
            prefetch_reorder_accuracy_percent=100 * combined['final_accuracy'],
            source_directory=str(directory.relative_to(ROOT.parent)),
            reused_from_table7=period == 105,
        ))
        table.append(f"{period} & {baseline['training_seconds']:.2f} & "
                     f"{combined['training_seconds']:.2f} & {saved:.2f} " + r'\\')

    savings = [r['training_time_saved_percent'] for r in comparisons]
    if min(savings) > 0:
        conclusion = (
            f'Reorder with prefetching reduces training time at all three periods, '
            f'by {min(savings):.2f}--{max(savings):.2f}\\% in these runs.\n'
            'The benefit therefore persists across the tested request periods.\n')
    else:
        positive = sum(x > 0 for x in savings)
        conclusion = (
            f'Reorder with prefetching reduces training time at {positive} of the three periods.\n'
            f'The measured reduction ranges from {min(savings):.2f}\\% to {max(savings):.2f}\\%; '
            'a negative value means longer training time.\n'
            'These results do not show a consistent reduction across all tested periods.\n')
    text = r'''% Generated from verified 12-epoch measurements; do not insert smoke timings.
% Provenance: ../orin_scheduling/period_sweep_protocol.txt
\begingroup
\color{blue}
\noindent\textbf{Varying the foreground inference period.}
We compare FIFO without caching and Reorder with prefetching at request periods of 105, 135, and 180\,ms.
Table~\ref{tab:orin-period-sweep} reports the time for all 12 training epochs, excluding evaluation, using the same settings as above.
The 105\,ms results are reused from Table~\ref{tab:orin-prefetch}; we run both methods once at each of the other two periods.
@@CONCLUSION@@

\begin{table}[!t]
\color{blue}
\caption{Training time under different foreground inference periods.}
\label{tab:orin-period-sweep}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.10}
\arrayrulecolor{blue}
\begin{tabular}{@{}rrrr@{}}
\toprule
\makecell{Period\\(ms)} & \makecell{FIFO\\(s)} & \makecell{Reorder +\\prefetch (s)} & \makecell{Training time\\saved (\%)} \\
\midrule
@@ROWS@@
\bottomrule
\end{tabular}
\arrayrulecolor{black}
\end{table}
\par
\endgroup
'''
    text = text.replace('@@CONCLUSION@@', conclusion).replace('@@ROWS@@', '\n'.join(table))
    assert '@@' not in text
    output = ROOT.parent / 'DynaProxyer_TMC/sec-eval-period-sweep.tex'
    output.write_text(text)
    (NEW / 'table8_comparisons.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    with (NEW / 'table8_metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print(output)
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
