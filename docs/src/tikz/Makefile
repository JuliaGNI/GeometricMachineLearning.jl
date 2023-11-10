pdf:
	xelatex -shell-escape commuting-diagram
	xelatex -shell-escape omega-orthogonal-projection
	xelatex -shell-escape post-projection
	xelatex -shell-escape post-projection-unstable
	xelatex -shell-escape standard-projection
	xelatex -shell-escape symmetric-projection-minus
	xelatex -shell-escape symmetric-projection-plus
	xelatex -shell-escape symplectic-projection-unstable
	xelatex -shell-escape variation-continuous
	xelatex -shell-escape variation-discrete

png:
	pdftocairo  -png -r 150 -transp -singlefile  commuting-diagram.pdf               commuting-diagram              
	pdftocairo  -png -r 150 -transp -singlefile  omega-orthogonal-projection.pdf     omega-orthogonal-projection    
	pdftocairo  -png -r 150 -transp -singlefile  post-projection.pdf                 post-projection                
	pdftocairo  -png -r 150 -transp -singlefile  post-projection-unstable.pdf        post-projection-unstable       
	pdftocairo  -png -r 150 -transp -singlefile  standard-projection.pdf             standard-projection            
	pdftocairo  -png -r 150 -transp -singlefile  symmetric-projection-minus.pdf      symmetric-projection-minus     
	pdftocairo  -png -r 150 -transp -singlefile  symmetric-projection-plus.pdf       symmetric-projection-plus      
	pdftocairo  -png -r 150 -transp -singlefile  symplectic-projection-unstable.pdf  symplectic-projection-unstable 
	pdftocairo  -png -r 150 -transp -singlefile  variation-continuous.pdf            variation-continuous           
	pdftocairo  -png -r 150 -transp -singlefile  variation-discrete.pdf              variation-discrete             

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

all: pdf