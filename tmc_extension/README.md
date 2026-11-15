# PETCube TMC extension

Current manuscript and the Jetson AGX Orin experiments for feature caching and foreground-aware reordering.

## Contents

- [Final manuscript PDF](DynaProxyer_TMC.pdf) and [LaTeX source](DynaProxyer_TMC/main.tex).
- `PieBridge/`: the original implementation plus the four benchmark/policy modules used for the formal Orin experiments.
- `orin_scheduling/`: experiment protocols, analysis/table generation scripts, and final results for Table 6 and Figure 18.

## Build the paper

From this directory, run:

```sh
bash DynaProxyer_TMC/build.sh
```

The output is `DynaProxyer_TMC/PieBridge_TMC.pdf`. The checked-in final PDF has 16 pages.

## Results included

Historical experiment files refer to Tables 7 and 8; these results now appear in Table 6 and Figure 18, respectively.

Table 6 uses `orin_scheduling/results/feature_20261113_prefetch_full_v2/`: four methods at 105 ms.
Figure 18 reuses the 105 ms pair and adds `feature_20261115_prefetch_periods_v1/`: FIFO and Reorder + prefetch at 135 and 180 ms.
Each condition runs once for 12 epochs, with full evaluation after every epoch. Reported training time excludes evaluation.
Feature caching is restricted to the current epoch; features are consumed once and released.
Figure 18 plots training time from a zero baseline and labels the percentage reduction relative to uncached FIFO. No error bars are shown because each condition was run once.

The result directories contain final aggregate metrics, per-epoch summaries, configuration, source snapshots, and the final verification outcome.
Published `main_p*.json` files retain all run/epoch metrics but omit `epoch_traces` and `inference_trace`; these detailed execution logs remain local. `publication_manifest.json` records the original and published hashes and the removed fields.
Smoke tests, intermediate experiments, progress snapshots, logs, build caches, downloaded runtimes, model/data assets, and old archives are not included.

To regenerate the manuscript result sections and the period-comparison chart from these summaries (the plot requires Python with Matplotlib, NumPy, and the Liberation Serif font):

```sh
python3 orin_scheduling/render_prefetch_section.py
python3 orin_scheduling/render_period_sweep.py
python3 DynaProxyer_TMC/figures/orin_period_sweep.py
```

To rerun experiments on a configured CUDA device, use `PieBridge/src/prefetch_training_benchmark.py` with the parameters in the corresponding `profile.json`; see the protocol files and the upstream PieBridge setup instructions for model/data assets.
`report_prefetch.py` verifies newly generated full raw outputs; it requires their detailed traces and is not intended for the trimmed published JSON summaries.
