all: directories images combined_patches mnist_visualization mnist_visualization_dark png

directories:
	mkdir -p final; mkdir -p flatten; mkdir -p original; mkdir -p split 

images:
	julia --project=../../.. mnist_visualization.jl
	
combined_patches:
	xelatex -shell-escape final_image.tex xelatex

mnist_visualization:
	xelatex -shell-escape mnist_visualization.tex

mnist_visualization_dark:
	xelatex -shell-escape mnist_visualization_dark.tex

clean: 
	rm -Rf final; \
	rm -Rf flatten; \
	rm -Rf original; \
	rm -Rf split; \
	rm *.pdf; \
	rm *.log; \
	rm *.aux; \
	rm *.png

png:
	pdftocairo  -png -r 120 -transp -singlefile  mnist_visualization.pdf        mnist_visualization  
	pdftocairo  -png -r 120 -transp -singlefile  mnist_visualization_dark.pdf   mnist_visualization_dark