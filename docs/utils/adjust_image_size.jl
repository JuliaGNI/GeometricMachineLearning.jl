file = open("build/GeometricMachineLearning.jl.tex")

collection_of_lines = Tuple(line for line in eachline(file));
close(file)

function adjust_image_size(path::AbstractString, size::String, lines::Union{Tuple, Vector}=collection_of_lines)
    new_contents = ""
    for line in lines
        if raw"\includegraphics[max width=\linewidth]{" * path * raw"}" == line
            new_contents *= raw"\includegraphics[width=" * size * raw"\textwidth]{" * path * raw"}"
            new_contents *= "\n"
        else
            new_contents *= line
            new_contents *= "\n"
        end
    end
    new_contents *= "\n"
end

new_contents = adjust_image_size(raw"tikz/tangent_vector_light.png", ".5", collection_of_lines)
new_contents = adjust_image_size(raw"manifolds/sphere_with_tangent_vec_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"manifolds/sphere_with_tangent_vec_and_geodesic_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"optimizers/manifold_related/parallel_transport_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"optimizers/manifold_related/two_vectors_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"optimizers/manifold_related/retraction_comparison_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"optimizers/manifold_related/retraction_discrepancy_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/sympnet_training_loss_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tikz/gml_venn_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tikz/symplectic_autoencoder_architecture_light.png", ".65", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/sae_validation_light.png", ".7", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/sae_integrator_validation_light.png", ".7", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/sympnet_prediction_light.png", ".55", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/sympnet_resnet_training_loss_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/resnet_sympnet_prediction_light.png", ".55", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tikz/symplectic_autoencoder_light.png", ".65", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tikz/sae_venn_light.png", ".35", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tikz/transformer_upscaling_light.png", ".6", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/resnet_sympnet_prediction_long_light.png", ".55", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/lst_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/lst_validation_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/rigid_body_trajectories_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/curve_comparison_light.png", ".6", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/training_loss_vpa_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/plot40_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/plot200_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/plot40_sine2_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/plot200_sine2_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tikz/third_degree_spline_light.png", ".3", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tikz/multiple_parameters_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tikz/linear_symplectic_transformer_light.png", ".3", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/plot40_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/training_loss2_vpa_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/plot400_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/rosenbrock_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/point_cloud_arrows_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/training_loss_light.png", ".4", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/mapped_points_light.png", ".5", split(new_contents, "\n"))
new_contents = adjust_image_size(raw"tutorials/compare_losses_light.png", ".5", split(new_contents, "\n"))

collection_of_lines = split(new_contents, "\n")
new_contents = ""
for (i, line) in zip(1:length(collection_of_lines), collection_of_lines)
    global new_contents *= replace(line, raw"\begin{figure}" => raw"\begin{figure}[H]")
    global new_contents *= "\n"
end

new_file = open("build/GeometricMachineLearning.jl-copy.tex", "a")
write(new_file, new_contents)
close(new_file)