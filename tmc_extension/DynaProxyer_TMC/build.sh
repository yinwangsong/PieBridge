#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
cp build/main.pdf PieBridge_TMC.pdf
