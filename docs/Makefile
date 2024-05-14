
.PHONY: documenter images

all: html

latex_no_images: latex_no_pdf_no_images
	$(MAKE) compile_tex;
	$(MAKE) compile_tex

latex: latex_no_pdf 
	$(MAKE) compile_tex;
	$(MAKE) compile_tex

latex_no_pdf_no_images: install_brenier_two_fluid latex_docs_no_pdf put_figures_outside_of_minted_environment do_correct_quotation_marks make_correct_thrm_and_dfntn_environment

latex_no_pdf: latex_images latex_no_pdf_no_images 

html: html_images html_no_images

html_no_images: install_brenier_two_fluid test_docs documenter

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
	$(MAKE) all -C src/tikz

latex_images: 
	$(MAKE) latex -C src/tikz

clean:
	$(MAKE) empty -C src/tikz ;
	rm -Rf build

latex_docs_no_pdf:
	julia --color=yes --project=. --threads=8 make.jl latex_output	

compile_tex: 
	cd build; \
	lualatex -shell-escape G*.tex;

put_figures_outside_of_minted_environment:
	sed -i'' -e 's/\"\\\\begin{figure}\\n\\\\includegraphics/DeleteThisAndTheLineBefore\n\\begin{figure}[H]\n\\centering\n\\includegraphics/g' build/G*.tex;
	sed -i'' -e 's/\\\\end{figure}\"/\\end{figure}\nDeleteThisAndTheLineAfter/g' build/G*.tex;
	sed -i'' -e 's/\\\\textwidth/\\textwidth/g' build/G*.tex;
	sed -i'' -e 's/\\\\label/\\label/g' build/G*.tex;
	sed -i'' -e 's/\\\\caption/\\caption/g' build/G*.tex;
	sed -i'' -e 's/..\/tikz/tikz/g' build/G*.tex;
	echo -en '\n\n' >> build/GeometricMachineLearning.jl.tex 
	sed -ni'' -e '/DeleteThisAndTheLineBefore/{x;d;};1h;1!{x;p;};${x;p;}' build/G*.tex;
	sed -i'' -e '/DeleteThisAndTheLineAfter/{N;s/\n.*//;}' build/G*.tex;
	sed -i'' -e '/DeleteThisAndTheLineBefore/d' build/G*.tex;
	sed -i'' -e '/DeleteThisAndTheLineAfter/d' build/G*.tex;

make_correct_thrm_and_dfntn_environment:
	sed -i'' -e 's/{\\textbackslash}begin\\{thrm\\}/\\begin{thrm}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}end\\{thrm\\}/\\end{thrm}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}label\\{th:\([a-zA-Z]*\)\\}/\\label{th:\1}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}begin\\{dfntn\\}/\\begin{dfntn}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}end\\{dfntn\\}/\\end{dfntn}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}label\\{def:\([a-zA-Z]*\)\\}/\\label{th:\1}/g' build/G*.tex;

do_correct_quotation_marks:
	sed -i'' -e 's/{\\textquotedbl}/"/g' build/G*.tex;
	sed -i'' -e 's/ "/ ``/g' build/G*.tex