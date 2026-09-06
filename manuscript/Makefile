# Makefile for compiling the IEEE TBME manuscript
MAIN = main
TEX = pdflatex
BIB = bibtex

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz

.PHONY: all clean
