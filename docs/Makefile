
.PHONY: documenter images

all: html

latex_mac: images_mac install_brenier_two_fluid test_docs latex_docs remove_brenier_two_fluid

latex_linux: images_linux install_brenier_two_fluid test_docs latex_docs remove_brenier_two_fluid

latex_linux_no_pdf: images_linux install_brenier_two_fluid test_docs latex_docs_no_pdf remove_brenier_two_fluid

latex_mac_no_pdf: images_mac install_brenier_two_fluid test_docs latex_docs_no_pdf remove_brenier_two_fluid

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

images_mac: 
	$(MAKE) mac		 -C src/tikz

images_linux: 
	$(MAKE) linux 	 -C src/tikz

clean:
	$(MAKE) empty -C src/tikz ;
	rm -Rf build

latex_docs:
	julia --color=yes --project=. --threads=8 make.jl latex_output

latex_docs_no_pdf:
	julia --color=yes --project=. --threads=8 make.jl latex_output_no_pdf	