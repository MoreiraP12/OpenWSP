# make: figures, tables and number macros from data/paper_v8.json, then the PDF
all: pdf

figures:
	python3 scripts/make_figures.py

pdf: figures
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -c
