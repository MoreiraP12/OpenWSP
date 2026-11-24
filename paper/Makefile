# make: figures, tables and number macros from data/paper_v8.json, then the PDF
# phony, or the figures/ directory would make the figures target look up to date and skip make_figures.py
.PHONY: all figures pdf clean
all: pdf

figures:
	python3 scripts/make_figures.py

pdf: figures
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -c
