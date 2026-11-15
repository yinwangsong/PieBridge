"""Package the current manuscript's local LaTeX inputs without build artifacts."""

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "DynaProxyer_TMC"
ARCHIVE = ROOT / "PETCube_TMC_LaTeX_Source.zip"
PREFIX = "PETCube_TMC_Source/"


def main():
    inputs = set()
    for name in ("main", "main-clean"):
        recorder = PAPER / "build" / f"{name}.fls"
        for line in recorder.read_text().splitlines():
            if not line.startswith("INPUT "):
                continue
            path = Path(line[6:])
            path = (path if path.is_absolute() else PAPER / path).resolve()
            if (path.is_file() and path.is_relative_to(PAPER)
                    and not path.is_relative_to(PAPER / "build")):
                inputs.add(path)
    for name in ("ref.bib", "IEEEtran.bst", "figures/orin_period_sweep.py",
                 "figures/orin_period_sweep_data.json"):
        inputs.add(PAPER / name)
    files = {p.relative_to(PAPER).as_posix(): p.read_bytes() for p in sorted(inputs)}
    files["cover_letter/cover_letter_tmc.tex"] = (
        ROOT / "submission/cover_letter_tmc.tex").read_bytes()
    files["README.txt"] = """PETCube — TMC LaTeX source package

本包对应当前 16 页 TMC 稿件，含所需正文、参考文献、图片和模板文件。
作者姓名为 Gang Huang；通讯作者为 Mengwei Xu。
Cover letter 位于 cover_letter/，落款为 Wangsong Yin。

编译入口
  main-clean.tex：正式稿，修订文字为黑色。
  main.tex：保留蓝色新增内容的标注稿。

Overleaf
  上传本 ZIP 新建项目，编译器选择 pdfLaTeX。
  Main document 选择 main-clean.tex（正式稿）或 main.tex（标注稿）。
  两个入口使用同一套正文、参考文献和图片。

Windows 本地编译
  安装包含 latexmk、pdfLaTeX、BibTeX 和常用宏包的 TeX Live。
  解压后双击 build.bat，同时编译正式稿与标注稿。
  输出 PETCube_TMC_clean.pdf、PETCube_TMC_marked.pdf。
  或在本目录的 PowerShell 中执行：
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main-clean.tex
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex

Linux / macOS
  在已安装 LaTeX 工具链的环境中执行 bash build.sh。

投稿信
  在 cover_letter 目录执行：
    latexmk -pdf -interaction=nonstopmode -halt-on-error cover_letter_tmc.tex

IEEEtran.cls、IEEEtran.bst、algorithm2e.sty 随包提供，保留上游文件内容。
figs/ 和 figures/ 中的 PDF 是正文所需的矢量图片，不是整篇论文预览。
orin_period_sweep.py 和对应 JSON 是最终柱状图的绘图源码与汇总数据。
不包含编译日志、缓存、旧版示例图、预览导出、实验原始记录、模型或数据集。
source_manifest.json 记录包内文件的 SHA-256。
""".encode("utf-8")
    files["build.bat"] = r"""@echo off
setlocal
cd /d "%~dp0"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main-clean.tex
if errorlevel 1 exit /b 1
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
if errorlevel 1 exit /b 1
copy /Y "build\main-clean.pdf" "PETCube_TMC_clean.pdf" >nul
if errorlevel 1 exit /b 1
copy /Y "build\main.pdf" "PETCube_TMC_marked.pdf" >nul
if errorlevel 1 exit /b 1
echo Built PETCube_TMC_clean.pdf and PETCube_TMC_marked.pdf
""".replace("\n", "\r\n").encode("utf-8")
    files["build.sh"] = """#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main-clean.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
cp build/main-clean.pdf PETCube_TMC_clean.pdf
cp build/main.pdf PETCube_TMC_marked.pdf
""".encode("utf-8")
    manifest = {"files": [dict(file=name, bytes=len(data),
                               sha256=hashlib.sha256(data).hexdigest())
                           for name, data in sorted(files.items())]}
    files["source_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(files.items()):
            z.writestr(PREFIX + name, data)
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for name, data in files.items():
            assert z.read(PREFIX + name) == data
        assert not any("/build/" in n or "/tavottofile/" in n or
                       n.endswith((".log", ".aux", ".bbl", ".blg", ".fls", ".out"))
                       for n in z.namelist())
    print(json.dumps(dict(archive=str(ARCHIVE), files=len(files),
                         bytes=ARCHIVE.stat().st_size,
                         sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()), indent=2))


if __name__ == "__main__":
    main()
