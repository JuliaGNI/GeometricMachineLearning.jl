
.PHONY: documenter images

all: html

latex: tex_images install_brenier_two_fluid test_docs latex_docs remove_brenier_two_fluid

html: html_images install_brenier_two_fluid test_docs documenter remove_brenier_two_fluid

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

html_images:
	$(MAKE) pdf   									-C src/tikz ;
	$(MAKE) png res1=150 res2=200 res3=100 res4=180 -C src/tikz	;
	$(MAKE) logo  									-C src/tikz ;
	$(MAKE) clean -C src/tikz

tex_images:
	$(MAKE) pdf   									-C src/tikz ;
	$(MAKE) png res1=450 res2=600 res3=300 res4=540 -C src/tikz ;
	$(MAKE) logo  									-C src/tikz	;								
	$(MAKE) clean -C src/tikz

clean:
	$(MAKE) empty -C src/tikz ;
	rm -Rf build

latex_docs:
	julia --color=yes --project=. --threads=8 make.jl latex_output
	