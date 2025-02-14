all: pdf 
	$(MAKE) png   res1=150 res2=200 res3=100 res4=180
	$(MAKE) logo 
	$(MAKE) clean 

latex: pdf 
	$(MAKE) convert_with_pdftocairo res=500
	$(MAKE) logo 
	$(MAKE) clean

MYDIR = .
pdf: $(MYDIR)/*.tex
	for file in $^ ; do \
		xelatex -shell-escape $${file} ; \
		xelatex -shell-escape $${file} ; \
	done

# this is converting pdfs to pngs using pdftocairo (with a fixed resolution for all images)
convert_with_pdftocairo: $(MYDIR)/*.pdf
	for file in $^ ; do \
		pdftocairo -png -r $(res) -transp -singlefile $${file} $${file%.*} ; \
	done

png:
	pdftocairo  -png -r $(res1) -transp -singlefile  adam_optimizer_light.pdf               	 	adam_optimizer_light
	pdftocairo  -png -r $(res1) -transp -singlefile  adam_optimizer_dark.pdf               	 		adam_optimizer_dark                     
	pdftocairo  -png -r $(res1) -transp -singlefile  general_optimization_light.pdf		 	 		general_optimization_light
	pdftocairo  -png -r $(res1) -transp -singlefile  general_optimization_with_boundary_light.pdf 	general_optimization_with_boundary_light              
	pdftocairo  -png -r $(res1) -transp -singlefile  general_optimization_with_boundary_dark.pdf	general_optimization_with_boundary_dark          
	pdftocairo  -png -r $(res1) -transp -singlefile  transformer_encoder_light.pdf		       		transformer_encoder_light   
	pdftocairo  -png -r $(res1) -transp -singlefile  transformer_encoder_dark.pdf		        	transformer_encoder_dark   
	pdftocairo  -png -r $(res1) -transp -singlefile  third_degree_spline_light.pdf	           		third_degree_spline_light
	pdftocairo  -png -r $(res1) -transp -singlefile  third_degree_spline_dark.pdf	           		third_degree_spline_dark       
	pdftocairo	-png -r $(res2) -transp -singlefile	 sympnet_architecture_light.pdf					sympnet_architecture_light
	pdftocairo	-png -r $(res2) -transp -singlefile	 sympnet_architecture_dark.pdf					sympnet_architecture_dark
	pdftocairo 	-png -r $(res3) -transp -singlefile  structs_visualization_light.pdf				structs_visualization_light
	pdftocairo 	-png -r $(res3) -transp -singlefile  structs_visualization_dark.pdf					structs_visualization_dark
	pdftocairo  -png -r 500     -transp -singlefile  logo_with_name_light.pdf                       logo_with_name_light
	pdftocairo  -png -r 500     -transp -singlefile  logo_with_name_dark.pdf                        logo_with_name_dark
	pdftocairo  -png -r 500     -transp -singlefile  logo_with_name_without_jl.pdf                  logo_with_name_without_jl
	pdftocairo 	-png -r $(res1) -transp -singlefile	 logo.pdf										logo 
	pdftocairo  -png -r $(res1) -transp -singlefile  symplectic_autoencoder_light.pdf				symplectic_autoencoder_light
	pdftocairo  -png -r $(res1) -transp -singlefile  symplectic_autoencoder_dark.pdf				symplectic_autoencoder_dark
	pdftocairo  -png -r $(res1) -transp -singlefile  symplectic_autoencoder_architecture_light.pdf	symplectic_autoencoder_architecture_light
	pdftocairo  -png -r $(res1) -transp -singlefile  symplectic_autoencoder_architecture_dark.pdf	symplectic_autoencoder_architecture_dark
	pdftocairo  -png -r $(res1) -transp -singlefile  solution_manifold_2.pdf						solution_manifold_2
	pdftocairo  -png -r $(res1) -transp -singlefile  mha_light.pdf 									mha_light
	pdftocairo  -png -r $(res1) -transp -singlefile  mha_dark.pdf 									mha_dark
	pdftocairo  -png -r $(res1) -transp -singlefile  tensor_light.pdf 								tensor_light
	pdftocairo  -png -r $(res1) -transp -singlefile  tensor_dark.pdf 								tensor_dark
	pdftocairo  -png -r $(res4) -transp -singlefile  tensor_sampling_light.pdf 						tensor_sampling_light
	pdftocairo  -png -r $(res4) -transp -singlefile  tensor_sampling_dark.pdf 						tensor_sampling_dark
	pdftocairo 	-png -r $(res1) -transp -singlefile  vp_feedforward_light.pdf						vp_feedforward_light
	pdftocairo 	-png -r $(res1) -transp -singlefile  vp_feedforward_dark.pdf						vp_feedforward_dark
	pdftocairo 	-png -r $(res1) -transp -singlefile  vp_transformer_light.pdf						vp_transformer_light
	pdftocairo 	-png -r $(res1) -transp -singlefile  vp_transformer_dark.pdf						vp_transformer_dark
	pdftocairo 	-png -r $(res1) -transp -singlefile  linear_symplectic_transformer_light.pdf		linear_symplectic_transformer_light
	pdftocairo 	-png -r $(res1) -transp -singlefile  linear_symplectic_transformer_dark.pdf			linear_symplectic_transformer_dark
	pdftocairo 	-png -r $(res1) -transp -singlefile  skew_sym_visualization_light.pdf				skew_sym_visualization_light
	pdftocairo 	-png -r $(res1) -transp -singlefile  skew_sym_visualization_dark.pdf				skew_sym_visualization_dark
	pdftocairo  -png -r $(res2) -transp -singlefile  offline_online_light.pdf						offline_online_light
	pdftocairo  -png -r $(res2) -transp -singlefile  offline_online_dark.pdf						offline_online_dark
	pdftocairo  -png -r $(res1) -transp -singlefile  ingredients_light.pdf							ingredients_light
	pdftocairo  -png -r $(res1) -transp -singlefile  ingredients_dark.pdf							ingredients_dark
	pdftocairo  -png -r $(res1) -transp -singlefile  convex_recombination_light.pdf					convex_recombination_light
	pdftocairo  -png -r $(res1) -transp -singlefile  convex_recombination_dark.pdf					convex_recombination_dark
	pdftocairo  -png -r $(res1) -transp -singlefile  skew_sym_mapping_light.pdf						skew_sym_mapping_light
	pdftocairo  -png -r $(res1) -transp -singlefile  skew_sym_mapping_dark.pdf						skew_sym_mapping_dark
	pdftocairo  -png -r $(res1) -transp -singlefile  sae_venn_light.pdf								sae_venn_light
	pdftocairo  -png -r $(res1) -transp -singlefile  sae_venn_dark.pdf								sae_venn_dark
	pdftocairo  -png -r $(res1) -transp -singlefile  structure_preservation_hierarchy_light.pdf		structure_preservation_hierarchy_light
	pdftocairo  -png -r $(res1) -transp -singlefile  structure_preservation_hierarchy_dark.pdf		structure_preservation_hierarchy_dark
	pdftocairo  -png -r $(res1) -transp -singlefile  transformer_upscaling_light.pdf				transformer_upscaling_light
	pdftocairo  -png -r $(res1) -transp -singlefile  transformer_upscaling_dark.pdf					transformer_upscaling_dark
	pdftocairo  -png -r $(res1) -transp -singlefile  multiple_parameters_light.pdf					multiple_parameters_light
	pdftocairo  -png -r $(res1) -transp -singlefile  multiple_parameters_dark.pdf					multiple_parameters_dark
	pdftocairo  -png -r $(res1) -transp -singlefile  coupled_harmonic_oscillator_light.pdf			coupled_harmonic_oscillator_light
	pdftocairo  -png -r $(res1) -transp -singlefile  coupled_harmonic_oscillator_dark.pdf			coupled_harmonic_oscillator_dark
	pdftocairo  -png -r $(res1) -transp -singlefile  tangent_vector_light.pdf						tangent_vector_light
	pdftocairo  -png -r $(res1) -transp -singlefile  tangent_vector_dark.pdf						tangent_vector_dark
	pdftocairo  -png -r $(res1) -transp -singlefile  gml_venn_light.pdf								gml_venn_light
	pdftocairo  -png -r $(res1) -transp -singlefile  gml_venn_dark.pdf								gml_venn_dark
	pdftocairo  -png -r $(res1) -transp -singlefile  reduced_order_modeling_idea_light.pdf			reduced_order_modeling_idea_light
	pdftocairo  -png -r $(res1) -transp -singlefile  reduced_order_modeling_idea_dark.pdf			reduced_order_modeling_idea_dark
	pdftocairo  -png -r $(res1) -transp -singlefile  grassmann_sampling_light.pdf					grassmann_sampling_light
	pdftocairo  -png -r $(res1) -transp -singlefile  grassmann_sampling_dark.pdf					grassmann_sampling_dark
	pdftocairo  -png -r $(res1) -transp -singlefile  lpH_equivalence_light.pdf						lpH_equivalence_light
	pdftocairo  -png -r $(res1) -transp -singlefile  lpH_equivalence_dark.pdf						lpH_equivalence_dark

logo:
	cp logo_with_name_light.png 		../assets/logo.png
	cp logo_with_name_dark.png 			../assets/logo-dark.png
	cp logo_with_name_without_jl.png 	../assets/logo_without_jl.png

clean:
	rm -f *.aux
	rm -f *.fdb_latexmk
	rm -f *.fls
	rm -f *.log
	rm -f *.xdv
	rm -f *.pdf

empty: clean
	rm -f *.png
	rm -f *.svg
	rm -f ../assets/*.png