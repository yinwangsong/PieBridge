"""Build and package the five PDFs for a TMC initial submission; never upload them."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "DynaProxyer_TMC"
SUBMISSION = ROOT / "submission"
OUTPUT = ROOT / "TMC_submission"


def run(command, cwd=ROOT):
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout


def pdf_info(path):
    info = run(["pdfinfo", str(path)])
    assert re.search(r"^Encrypted:\s+no\s*$", info, re.M), path
    return int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))


def main():
    conference = SUBMISSION / "supporting/PieBridge_SenSys2024.pdf"
    assert conference.is_file(), "The author-hosted SenSys PDF is required."
    run(["bash", "build.sh"], cwd=PAPER)
    run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         "-file-line-error", "-outdir=build", "main-clean.tex"], cwd=PAPER)
    run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         "-outdir=build", "cover_letter_tmc.tex"], cwd=SUBMISSION)
    run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         "-outdir=build", "extension_statement_tmc.tex"], cwd=SUBMISSION)
    for log in (PAPER / "build/main.log", PAPER / "build/main-clean.log",
                SUBMISSION / "build/cover_letter_tmc.log",
                SUBMISSION / "build/extension_statement_tmc.log"):
        content = log.read_text()
        assert not any(term in content for term in ("Warning", "Overfull", "undefined")), log
    shutil.copyfile(SUBMISSION / "build/cover_letter_tmc.pdf", SUBMISSION / "cover_letter_tmc.pdf")
    shutil.copyfile(SUBMISSION / "build/extension_statement_tmc.pdf", SUBMISSION / "extension_statement_tmc.pdf")

    clean = PAPER / "build/main-clean.pdf"
    marked = PAPER / "PieBridge_TMC.pdf"
    clean_text = run(["pdftotext", str(clean), "-"])
    assert clean_text == run(["pdftotext", str(marked), "-"]), "Clean/marked text differs."
    assert "This article extends our SenSys 2024 paper" in clean_text
    assert "Gang Huang" in clean_text and "Gang Hang" not in clean_text
    assert "\\bibitem{yin2024piebridge}" in (PAPER / "build/main-clean.bbl").read_text()
    assert "10.1145/3666025.3699327" in (PAPER / "ref.bib").read_text()
    assert pdf_info(clean) == pdf_info(marked) == 16
    assert pdf_info(conference) == 15
    assert pdf_info(SUBMISSION / "cover_letter_tmc.pdf") == 1
    assert pdf_info(SUBMISSION / "extension_statement_tmc.pdf") == 2
    conference_text = run(["pdftotext", "-f", "1", "-l", "1", str(conference), "-"])
    assert "PieBridge" in conference_text and "3666025.3699327" in conference_text

    OUTPUT.mkdir(exist_ok=True)
    documents = [
        ("01_PETCube_Manuscript.pdf", clean, "main_manuscript"),
        ("02_PETCube_Cover_Letter.pdf", SUBMISSION / "cover_letter_tmc.pdf", "cover_letter"),
        ("03_PieBridge_SenSys2024.pdf", conference, "previously_published_paper"),
        ("04_PETCube_Marked_Manuscript.pdf", marked, "marked_supporting_manuscript"),
        ("05_PETCube_Extension_Statement.pdf", SUBMISSION / "extension_statement_tmc.pdf", "conference_extension_statement"),
    ]
    entries = []
    for name, source, role in documents:
        target = OUTPUT / name
        shutil.copyfile(source, target)
        data = target.read_bytes()
        entries.append(dict(file=name, role=role, pages=pdf_info(target), bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest()))
    guide_name = "00_UPLOAD_GUIDE.txt"
    shutil.copyfile(SUBMISSION / "upload_guide.txt", OUTPUT / guide_name)
    manifest = dict(
        journal="IEEE Transactions on Mobile Computing",
        document_type="Regular paper; initial submission package",
        submitted_to_journal=False,
        cover_letter_signatory="Wangsong Yin",
        corresponding_author="Mengwei Xu",
        clean_and_marked_text_identical=True,
        author_declarations_confirmed=False,
        conference_doi="10.1145/3666025.3699327",
        conference_source="https://xumengwei.github.io/files/SenSys24-PieBridge.pdf",
        conference_download="https://raw.githubusercontent.com/xumengwei/xumengwei.github.io/master/files/SenSys24-PieBridge.pdf",
        files=entries,
    )
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    names = [entry[0] for entry in documents] + [guide_name, "manifest.json"]
    assert set(p.name for p in OUTPUT.iterdir()) == set(names), "Unexpected file in submission directory."
    archive = ROOT / "TMC_submission.zip"
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in names:
            z.write(OUTPUT / name, "TMC_submission/" + name)
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        for name in names:
            assert z.read("TMC_submission/" + name) == (OUTPUT / name).read_bytes()
    temporary.replace(archive)
    print(json.dumps(dict(output=str(OUTPUT), archive=str(archive), files=entries), indent=2))


if __name__ == "__main__":
    main()
