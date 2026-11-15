PETCube TMC manuscript
======================
Main source: main.tex. New journal content is marked in blue.
Build from tmc_extension/: bash DynaProxyer_TMC/build.sh
Requires a LaTeX environment with latexmk and pdflatex.
The build produces DynaProxyer_TMC/PieBridge_TMC.pdf.
The published final PDF is ../DynaProxyer_TMC.pdf (16 pages).
Table 6 is a single-column table on page 12; Figure 18 is on page 13. Experiment summaries and source code are in
../orin_scheduling/ and ../PieBridge/.

TMC submission bundle (2026-11-15)
- Author spelling corrected to Gang Huang with author confirmation.
- Introduction cites PieBridge (SenSys 2024) and identifies the journal extensions.
- main.tex retains blue markings; main-clean.tex suppresses revision markings.
  Both PDFs have 16 pages and identical text.
- Run python3 submission/prepare_submission.py from tmc_extension/ to build the
  four-PDF bundle. See ../TMC_submission/00_UPLOAD_GUIDE.txt for upload roles.
