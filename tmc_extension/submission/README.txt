TMC cover letter draft
======================
cover_letter_tmc.txt: editable plain text for copying into a submission form.
cover_letter_tmc.tex: editable LaTeX source.
cover_letter_tmc.pdf: compiled letter.
extension_statement_tmc.tex: editable statement comparing the TMC and SenSys 2024 versions.
extension_statement_tmc.pdf: two-page statement for the editor and reviewers.

Build from the project root:
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=submission/build submission/cover_letter_tmc.tex
  cp submission/build/cover_letter_tmc.pdf submission/cover_letter_tmc.pdf

The letter uses the current manuscript title and is signed by Wangsong Yin,
using his affiliation and email from the manuscript. It explicitly discloses
the SenSys 2024 version and the journal additions.
Its date is the drafting date; update it if submitting on a later date.

Author confirmations pending:
- The manuscript is not under review by another journal or conference.
- All authors approve submission to TMC.
These declarations are omitted until confirmed; no submission has been made.

The manuscript now cites the SenSys paper (DOI 10.1145/3666025.3699327)
and describes the added material at the end of the Introduction.
supporting/PieBridge_SenSys2024.pdf is the unchanged author-hosted conference PDF.
The cover letter accompanies a clean main manuscript and a separate blue-marked copy.

Prepare the five-PDF submission bundle from the project root:
  python3 submission/prepare_submission.py
Output: TMC_submission/ and TMC_submission.zip (local delivery archive).
The bundle includes an upload guide; it excludes previews, build files, and logs.
The fifth PDF, 05_PETCube_Extension_Statement.pdf, separates the conference
contributions from the new online designs and Orin experiments, with section,
figure, table, and page references to the current 16-page manuscript.
No files are submitted to the journal by this command.

Sources checked on 15 November 2026:
IEEE Computer Society author guidelines (prior-publication disclosure):
  https://www.computer.org/publications/author-resources
TMC scope and call for papers:
  https://www.computer.org/digital-library/journals/tm/cfp-ieee-transactions-mobile-computing
SenSys 2024 paper:
  https://doi.org/10.1145/3666025.3699327
  https://xumengwei.github.io/files/SenSys24-PieBridge.pdf
