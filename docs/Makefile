
.PHONY: documenter images

all: html

latex_no_images: latex_no_pdf_no_images
	$(MAKE) compile_tex;
	$(MAKE) compile_tex

latex: latex_no_pdf 
	$(MAKE) compile_tex;
	$(MAKE) compile_tex

latex_no_pdf_no_images: install_brenier_two_fluid latex_docs_no_pdf copy_png_files put_figures_outside_of_minted_environment do_correct_quotation_marks make_correct_thrm_and_dfntn_and_xmpl_and_rmrk_and_proof_environment references_instead_of_hyperlinks correct_appendix remove_home_part remove_numbering_for_some_chapters

latex_no_pdf: latex_images mnist_image latex_no_pdf_no_images

html: html_images mnist_image html_no_images

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

mnist_image:
	$(MAKE) all -C src/tutorials/mnist

clean:
	$(MAKE) empty -C src/tikz ;
	$(MAKE) clean -C src/tutorials/mnist ;
	rm -Rf build

latex_docs_no_pdf:
	julia --color=yes --project=. --threads=8 make.jl latex_output	

compile_tex: 
	cd build; \
	xelatex -shell-escape G*.tex;

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
	sed -i'' -e 's/\\\\texttt/\\texttt/g' build/G*.tex;
	sed -i'' -e 's/\\\\_/\\_/g' build/G*.tex;

make_correct_thrm_and_dfntn_and_xmpl_and_rmrk_and_proof_environment:
	sed -i'' -e 's/{\\textbackslash}begin\\{thrm\\}/\\begin{thrm}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}end\\{thrm\\}/\\end{thrm}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}label\\{th:\([a-zA-Z]*\)\\}/\\label{th:\1}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}begin\\{dfntn\\}/\\begin{dfntn}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}end\\{dfntn\\}/\\end{dfntn}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}label\\{def:\([a-zA-Z]*\)\\}/\\label{def:\1}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}begin\\{xmpl\\}/\\begin{xmpl}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}end\\{xmpl\\}/\\end{xmpl}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}label\\{xmpl:\([a-zA-Z]*\)\\}/\\label{xmpl:\1}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}begin\\{proof\\}/\\begin{proof}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}end\\{proof\\}/\\end{proof}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}begin\\{rmrk\\}/\\begin{rmrk}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}end\\{rmrk\\}/\\end{rmrk}/g' build/G*.tex;
	sed -i'' -e 's/{\\textbackslash}label\\{rmrk:\([a-zA-Z]*\)\\}/\\label{rmrk:\1}/g' build/G*.tex;

do_correct_quotation_marks:
	sed -i'' -e 's/{\\textquotedbl}/"/g' build/G*.tex;
	sed -i'' -e 's/ "/ ``/g' build/G*.tex

references_instead_of_hyperlinks:
	sed -i'' -e 's/\\hyperlinkref{\([0-9]*\)}{\([a-zA-Z -]*\)}/\2 (see \\Cref{\1})/g' build/G*.tex

correct_appendix:
	sed -i'' -e 's/\\part{Appendix}/\\appendix\\part{Appendix}/g' build/G*.tex

remove_home_part:
	sed -i'' -e 's/\\part{HOME}/ /g' build/G*.tex

remove_numbering_for_some_chapters:
	sed -i'' -e 's/\\chapter{References}/\\chapter*{References}/g' build/G*.tex;
	sed -i'' -e 's/\\chapter{Preliminary Remarks and Acknowledgements}/\\chapter*{Preliminary Remarks and Acknowledgements}/g' build/G*.tex;
	sed -i'' -e 's/\\chapter{Abstract}/\\chapter*{Abstract}/g' build/G*.tex;
	sed -i'' -e 's/\\chapter{Zusammenfassung}/\\chapter*{Zusammenfassung}/g' build/G*.tex;


copy_png_files:
	find build/manifolds -name \*.png -exec cp {} build \; ;
	find build/optimizers/manifold_related -name \*.png -exec cp {} build \; ;
	find build/tutorials -name \*.png -exec cp {} build \;
	find build/reduced_order_modeling -name \*.png -exec cp {} build \;
	find build/tutorials/mnist -name \*.png -exec cp {} build \;