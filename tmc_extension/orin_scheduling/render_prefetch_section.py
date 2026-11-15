"""Render the blue manuscript extension after complete, verified measurements."""
import csv
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    data = root/'results/feature_20261113_prefetch_full_v2'
    assert json.loads((data/'status.json').read_text())['state'] == 'complete'
    assert (data/'local_verification.txt').read_text().startswith('PASS:')
    rows = {r['mode']:r for r in csv.DictReader((data/'metrics.csv').open())}
    labels = {'fifo':'FIFO', 'reorder':'Reorder',
              'prefetch_fifo':'FIFO + prefetch', 'prefetch_reorder':'Reorder + prefetch'}
    assert set(rows) == set(labels)
    assert all(float(r['period_ms']) == 105 and int(r['epochs']) == 12 for r in rows.values())
    baseline = rows['fifo']
    table = []
    for mode, label in labels.items():
        r = rows[mode]
        saved = 100*(1-float(r['training_seconds'])/float(baseline['training_seconds']))
        table.append(f"{label} & {float(r['training_seconds']):.2f} & "
                     f"{saved:.2f} & {float(r['accuracy_percent']):.2f} & "
                     f"{float(r['inference_p99_ms']):.2f} & "
                     f"{float(r['deadline_miss_percent']):.4f} " + r'\\')
    comparisons = {r['mode']:r for r in json.loads((data/'comparisons.json').read_text())}
    a, b = comparisons['prefetch_fifo'], comparisons['prefetch_reorder']
    extra = 100*(1-float(rows['prefetch_reorder']['training_seconds'])/float(rows['prefetch_fifo']['training_seconds']))
    extra_verb = 'reduces' if extra >= 0 else 'increases'
    peak = max(float(r['peak_feature_buffer_bytes']) for r in rows.values())/1024
    text = r'''% Generated only from complete, verified full-12-epoch measurements.
% Evidence: ../orin_scheduling/results/feature_20261113_prefetch_full_v2/
\begingroup
\color{blue}
\noindent\textbf{Prefetching.}
We compute features for upcoming batches before the classifier needs them, as described in Section~\ref{subsec:feature-prefetch}.
We look ahead by up to 128 batches and reserve space for at most 32 batches of features.
A single forward call can process one, two, or four batches that use the same proxy.
The classifier still uses batches of four images and updates in the original order in both prefetch methods.
Each selected training image passes through its assigned proxy once per epoch.
We use each feature once and then release it; the next epoch computes new features.
Evaluation also recomputes its features.
All prefetch work counts toward training time, from preparing the first features to using the last ones.

\begin{table}[!t]
\color{blue}
\caption{Experiment on Jetson orin with a periodic forground inference workload.}
\label{tab:orin-prefetch}
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.10}
\arrayrulecolor{blue}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Method & \makecell{Training\\time (s)} & \makecell{Training\\time saved\\(\%)} & \makecell{Accuracy\\(\%)} & \makecell{Inference\\p99 (ms)} & \makecell{Misses\\(\%)} \\
\midrule
@@ROWS@@
\bottomrule
\end{tabular}
\arrayrulecolor{black}
\end{table}

\noindent\textbf{Results.}
@@RESULTS@@

\par
\endgroup
'''
    results = (
        f"Table~\\ref{{tab:orin-prefetch}} shows the results. Accuracy is measured after epoch 12. "
        f"A miss means that an inference request takes more than 33.20\\,ms to finish.\n"
        f"Compared with FIFO without caching, Reorder alone cuts training time by {comparisons['reorder']['training_time_saved_percent']:.2f}\\%.\n"
        f"Adding prefetching to FIFO cuts it by {a['training_time_saved_percent']:.2f}\\%, "
        f"and combining Reorder with prefetching cuts it by {b['training_time_saved_percent']:.2f}\\%.\n"
        f"On top of FIFO with prefetching, reordering feature computation {extra_verb} training time by a further {abs(extra):.2f}\\%.\n\n"
        f"Final accuracy is {float(baseline['accuracy_percent']):.2f}\\% for FIFO without caching, "
        f"{float(rows['prefetch_fifo']['accuracy_percent']):.2f}\\% for FIFO with prefetching, and "
        f"{float(rows['prefetch_reorder']['accuracy_percent']):.2f}\\% for Reorder with prefetching.\n"
        f"The combined method is {abs(b['accuracy_difference_pp']):.2f} percentage points "
        f"{'below' if b['accuracy_difference_pp'] < 0 else 'above'} the baseline. "
        f"These runs do not establish that the methods reach the same final accuracy.\n"
        f"The feature buffer uses at most {peak:.2f}\\,KiB in these runs. "
        f"This excludes memory for models, inputs, and operator workspaces.\n"
    )
    text = text.replace('@@ROWS@@','\n'.join(table)).replace('@@RESULTS@@',results)
    assert '@@' not in text
    setup = (root/'paper_full12_template.tex').read_text()
    assert '@@' not in setup
    # Declare the single-column table near the experiment setup.
    start = text.index(r'\begin{table}')
    end = text.index(r'\end{table}', start) + len(r'\end{table}')
    table_float = text[start:end]
    text = text[:start] + text[end:]
    anchor = r'\label{subsec:orin-periodic}'
    assert setup.count(anchor) == 1
    setup = setup.replace(anchor, anchor + '\n\n' + table_float)
    setup_target = root.parent/'DynaProxyer_TMC/sec-eval-orin.tex'
    setup_target.write_text(setup)
    target = root.parent/'DynaProxyer_TMC/sec-eval-prefetch.tex'
    target.write_text(text)
    print(setup_target)
    print(target)


if __name__ == '__main__':
    main()
