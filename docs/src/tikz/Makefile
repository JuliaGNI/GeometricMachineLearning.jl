all: pdf png logo clean 

linux: pdf pdftocairo logo clean

mac: pdf sips logo clean

MYDIR = .
pdf: $(MYDIR)/*.tex
	for file in $^ ; do \
		xelatex -shell-escape $${file} ; \
	done

# this is converting pdfs to pngs using sips (mac version)
sips: $(MYDIR)/*.pdf
	for file in $^ ; do \
		sips --setProperty format png --resampleHeightWidthMax 2000 $${file} --out $${file%.*}.png ; \
	done

pdftocairo: $(MYDIR)/*.pdf
	for file in $^ ; do \
		pdftocairo -png -r 500 -transp -singlefile $${file} --out $${file%.*} ; \
	done

png:
	pdftocairo  -png -r 150 -transp -singlefile  adam_optimizer.pdf               	 			adam_optimizer  
	pdftocairo  -png -r 150 -transp -singlefile  adam_optimizer_dark.pdf               	 		adam_optimizer_dark                     
	pdftocairo  -png -r 150 -transp -singlefile  general_optimization.pdf		 	 			general_optimization   
	pdftocairo  -png -r 150 -transp -singlefile  general_optimization_with_boundary.pdf 		general_optimization_with_boundary              
	pdftocairo  -png -r 150 -transp -singlefile  general_optimization_with_boundary_dark.pdf	general_optimization_with_boundary_dark          
	pdftocairo  -png -r 150 -transp -singlefile  transformer_encoder.pdf		        		transformer_encoder   
	pdftocairo  -png -r 150 -transp -singlefile  third_degree_spline.pdf	           			third_degree_spline     
	pdftocairo  -png -r 150 -transp -singlefile  third_degree_spline_dark.pdf	           		third_degree_spline_dark       
	pdftocairo	-png -r 200 -transp -singlefile	 sympnet_architecture.pdf						sympnet_architecture
	pdftocairo	-png -r 200 -transp -singlefile	 sympnet_architecture_dark.pdf					sympnet_architecture_dark
	pdftocairo 	-png -r 100 -transp -singlefile  structs_visualization.pdf						structs_visualization
	pdftocairo 	-png -r 100 -transp -singlefile  structs_visualization_dark.pdf					structs_visualization_dark
	pdftocairo 	-png -r 150 -transp -singlefile	 logo.pdf										logo 
	pdftocairo  -png -r 150 -transp -singlefile  symplectic_autoencoder.pdf						symplectic_autoencoder
	pdftocairo  -png -r 150 -transp -singlefile  symplectic_autoencoder_dark.pdf				symplectic_autoencoder_dark
	pdftocairo  -png -r 150 -transp -singlefile  solution_manifold_2.pdf						solution_manifold_2
	pdftocairo  -png -r 150 -transp -singlefile  mha.pdf 										mha
	pdftocairo  -png -r 150 -transp -singlefile  mha_dark.pdf 									mha_dark
	pdftocairo  -png -r 150 -transp -singlefile  tensor.pdf 									tensor
	pdftocairo  -png -r 150 -transp -singlefile  tensor_dark.pdf 								tensor_dark
	pdftocairo  -png -r 180 -transp -singlefile  tensor_sampling.pdf 							tensor_sampling
	pdftocairo  -png -r 180 -transp -singlefile  tensor_sampling_dark.pdf 						tensor_sampling_dark
	pdftocairo 	-png -r 500 -transp -singlefile	 logo_with_name.pdf								logo_with_name 
	pdftocairo 	-png -r 500 -transp -singlefile  logo_with_name_dark.pdf 						logo_with_name_dark

logo:
	cp logo_with_name.png 		../assets/logo.png
	cp logo_with_name_dark.png 	../assets/logo-dark.png

clean:
	rm -f *.aux
	rm -f *.fdb_latexmk
	rm -f *.fls
	rm -f *.log
	rm -f *.pdf
	rm -f *.xdv

empty: clean
	rm -f *.png
	rm -f *.svg
	rm -f ../assets/*.png