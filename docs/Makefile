
.PHONY: documenter images

all: images documenter

latex: pdfs latex_docs

documenter:
	julia --color=yes --project=. --threads=14 make.jl html_output

images:
	$(MAKE) all   	 -C src/tikz

pdfs:
	$(MAKE) pdf_only -C src/tikz

clean:
	$(MAKE) empty 	 -C src/tikz
	rm -Rf build

latex_docs:
	julia --color=yes --project=. --threads=14 make.jl latex_output
	