TMC cover letter draft
======================
cover_letter_tmc.txt: editable plain text for copying into a submission form.
cover_letter_tmc.tex: editable LaTeX source.
cover_letter_tmc.pdf: compiled letter.

Build from the project root:
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=submission/build submission/cover_letter_tmc.tex
  cp submission/build/cover_letter_tmc.pdf submission/cover_letter_tmc.pdf

The letter uses the current manuscript title and corresponding-author details,
and explicitly discloses the SenSys 2024 version and the journal additions.
Its date is the drafting date; update it if submitting on a later date.

Author confirmations pending:
- The manuscript is not under review by another journal or conference.
- All authors approve submission to TMC.
These declarations are omitted until confirmed; no submission has been made.

Before submission, cite the SenSys paper in the manuscript and provide a copy
as supporting documentation. The current manuscript has no explicit citation
to DOI 10.1145/3666025.3699327. This task drafts the cover letter only.

Sources checked on 15 November 2026:
IEEE Computer Society author guidelines (prior-publication disclosure):
  https://www.computer.org/publications/author-resources
TMC scope and call for papers:
  https://www.computer.org/digital-library/journals/tm/cfp-ieee-transactions-mobile-computing
SenSys 2024 paper:
  https://doi.org/10.1145/3666025.3699327
  https://xumengwei.github.io/files/SenSys24-PieBridge.pdf
