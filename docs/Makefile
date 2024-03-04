
.PHONY: documenter images

all: images documenter

latex: images latex_docs

documenter:
	julia --color=yes --threads=14 --project=. make.jl html_output

images:
	$(MAKE) all   -C src/tikz

clean:
	$(MAKE) empty -C src/tikz
	rm -Rf build

latex_docs: 
	julia --color=yes --threads=14 --project=. make.jl latex_output