# PETCube TMC extension

Current manuscript and the Jetson AGX Orin experiments for feature caching and foreground-aware reordering.

## Contents

- [Final manuscript PDF](DynaProxyer_TMC.pdf) and [LaTeX source](DynaProxyer_TMC/main.tex).
- [TMC cover letter draft](submission/cover_letter_tmc.pdf), [editable text](submission/cover_letter_tmc.txt), and [submission notes](submission/README.txt).
- `PieBridge/`: the original implementation plus the four benchmark/policy modules used for the formal Orin experiments.
- `orin_scheduling/`: experiment protocols, analysis/table generation scripts, and final results for Table 6 and Figure 18.

## Initial submission files

The [submission bundle](TMC_submission/00_UPLOAD_GUIDE.txt) contains five PDFs: a clean main manuscript (16 pages), a cover letter signed by Wangsong Yin (1 page), the original SenSys 2024 paper (15 pages), a separate blue-marked manuscript (16 pages), and a [statement of differences from the conference paper](TMC_submission/05_PETCube_Extension_Statement.pdf) (2 pages). The main and marked copies have identical text and pagination. The Introduction cites the conference version and describes the extensions.

Run `python3 submission/prepare_submission.py` to rebuild this bundle and a local `TMC_submission.zip`. Upload the PDFs individually by their roles; the ZIP is for local delivery. No submission to the journal is performed.

## Download the LaTeX source

[Download the LaTeX ZIP](https://raw.githubusercontent.com/yinwangsong/PieBridge/tmc/tmc_extension/PETCube_TMC_LaTeX_Source.zip) includes the current manuscript, bibliography, required figures, IEEE template files, cover-letter source, and extension-statement source. It excludes build outputs, logs, previews, and experimental traces.

After extraction, select `main-clean.tex` for the clean submission or `main.tex` for the blue-marked version. The package includes Windows `build.bat`, Linux/macOS `build.sh`, and instructions for Overleaf. Both manuscript entry points reproduce the 16-page submission text; the cover letter builds to one page and the extension statement to two pages.

To regenerate the source ZIP, first build both manuscript entry points (for example with `python3 submission/prepare_submission.py`), then run `python3 submission/package_latex_source.py`. The script selects the actual local LaTeX inputs from the compiler recorder files.

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
Table 6 uses a single-column layout; Figure 18 is included at 72 mm width to match the surrounding text visually.
Figure 18 plots training time from a zero baseline and labels the percentage reduction relative to uncached FIFO. No error bars are shown because each condition was run once.

The result directories contain final aggregate metrics, per-epoch summaries, configuration, source snapshots, and the final verification outcome.
Published `main_p*.json` files retain all run/epoch metrics but omit `epoch_traces` and `inference_trace`; these detailed execution logs remain local. `publication_manifest.json` records the original and published hashes and the removed fields.
Smoke tests, intermediate experiments, progress snapshots, logs, build caches, downloaded runtimes, model/data assets, and old archives are not included.

To regenerate the manuscript result sections and the period-comparison chart from these summaries (the plot requires Python with Matplotlib, NumPy, and the DejaVu Sans font):

```sh
python3 orin_scheduling/render_prefetch_section.py
python3 orin_scheduling/render_period_sweep.py
python3 DynaProxyer_TMC/figures/orin_period_sweep.py
```

To rerun experiments on a configured CUDA device, use `PieBridge/src/prefetch_training_benchmark.py` with the parameters in the corresponding `profile.json`; see the protocol files and the upstream PieBridge setup instructions for model/data assets.
`report_prefetch.py` verifies newly generated full raw outputs; it requires their detailed traces and is not intended for the trimmed published JSON summaries.
