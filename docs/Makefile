
.PHONY: documenter images

all: images documenter

documenter:
	julia --color=yes --project=. make.jl

images:
	$(MAKE) all   -C src/images
	$(MAKE) clean -C src/images

clean:
	$(MAKE) empty -C src/images
	rm -Rf build
	rm -Rf src/tutorial