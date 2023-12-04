
.PHONY: documenter images

all: images documenter

documenter:
	julia --color=yes --project=. make.jl

images:
	$(MAKE) all   -C src/tikz

clean:
	$(MAKE) empty -C src/tikz
	rm -Rf build
	rm -Rf src/tutorial