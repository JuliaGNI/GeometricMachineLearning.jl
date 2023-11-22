pdf:
	xelatex -shell-escape adam_optimizer
	xelatex -shell-escape general_optimization
	xelatex -shell-escape general_optimization_with_boundary
	xelatex -shell-escape transformer_encoder
	xelatex -shell-escape third_degree_spline
	xelatex -shell-escape sympnet_architecture
	xelatex -shell-escape structs_visualization

png:
	pdftocairo  -png -r 150 -transp -singlefile  adam_optimizer.pdf               	 	adam_optimizer              
	pdftocairo  -png -r 150 -transp -singlefile  general_optimization.pdf		 	 	general_optimization   
	pdftocairo  -png -r 150 -transp -singlefile  general_optimization_with_boundary.pdf general_optimization_with_boundary              
	pdftocairo  -png -r 150 -transp -singlefile  transformer_encoder.pdf		        transformer_encoder   
	pdftocairo  -png -r 150 -transp -singlefile  third_degree_spline.pdf	           	third_degree_spline          
	pdftocairo	-png -r 150 -transp -singlefile	 sympnet_architecture.pdf				sympnet_architecture
	pdftocairo 	-png -r 150 -transp -singlefile  structs_visualization.pdf				structs_visualization

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

all: pdf png clean