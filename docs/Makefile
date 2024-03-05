
.PHONY: documenter images

all: html

latex: images install_brenier_two_fluid test_docs latex_docs remove_brenier_two_fluid

html: images install_brenier_two_fluid test_docs documenter remove_brenier_two_fluid

test_docs: 
	cd ..; julia --project=docs  -e ';									\
		using Pkg;														\
		Pkg.develop(PackageSpec(path=pwd()));							\
		Pkg.instantiate();												\
		Pkg.build();													\
		Pkg.precompile();												\
		using Documenter: doctest;										\
		using GeometricMachineLearning;									\
		doctest(GeometricMachineLearning)'					

install_brenier_two_fluid:
	cd ..; julia --project=docs -e ';									\
		using Pkg;														\
		Pkg.add(url="https://github.com/ToBlick/BrenierTwoFluids.git")'

remove_brenier_two_fluid:
	cd ..; julia --project=docs -e ';									\
		using Pkg;														\
		Pkg.rm("BrenierTwoFluid")'

documenter:
	julia --color=yes --project=. --threads=8 make.jl html_output

images:
	$(MAKE) all   	 -C src/tikz

clean:
	$(MAKE) empty 	 -C src/tikz
	rm -Rf build

latex_docs:
	julia --color=yes --project=. --threads=8 make.jl latex_output
	