
.PHONY: documenter images

all: html

latex: images latex_docs

html: images documenter

documenter:
	julia --color=yes --project=. --threads=8 make.jl html_output

images:
	$(MAKE) all   	 -C src/tikz

clean:
	$(MAKE) empty 	 -C src/tikz
	rm -Rf build

latex_docs:
	julia --color=yes --project=. --threads=8 make.jl latex_output
	