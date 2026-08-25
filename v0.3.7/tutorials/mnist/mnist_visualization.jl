using GeometricMachineLearning
using CairoMakie
import MLDatasets

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

#MNIST images are 28×28, so a sequence_length of 16 = 4² means the image patches are of size 7² = 49
image_dim = 28
patch_length = 7
n_heads = 7
n_layers = 5
patch_number = (image_dim÷patch_length)^2

train_x, train_y = MLDatasets.MNIST(split=:train)[:]

#preprocessing steps 
first_image = train_x[:, :, 8]

function split_image(image::AbstractMatrix, pl)
    n, m = size(image)
    @assert n == m
    @assert n%pl == 0
    #square root of patch number
    pnsq = n ÷ pl
    small_images = Tuple(map(i -> zeros(eltype(image), pl, pl), 1:pnsq^2))
    for i in 1:pnsq
        for j in 1:pnsq
            small_images[pnsq*(j-1)+i] .= image[pl*(i-1)+1:pl*i,pl*(j-1)+1:pl*j]
        end 
    end
    #Tuple(vcat(map(j -> map(i -> image[pl*(i-1)+1:pl*i,pl*(j-1)+1:pl*j,1]), 1:pnsq),1:pnsq)...)
    small_images
end

processed_image₁ = split_image(first_image, patch_length)
processed_image₂ = Tuple(map(i -> reshape(processed_image₁[i], 49, 1), 1:16))

fully_processed_image = split_and_flatten(first_image; patch_length = patch_length, number_of_patches = patch_number)

#see https://github.com/JuliaImages/ImageView.jl/issues/28
function _write_to_png(pic::AbstractMatrix, filename)
    first_axis, second_axis = axes(pic)
    fig = Figure(; backgroundcolor = :transparent)
    ax = Axis(fig[1, 1], 
                        backgroundcolor = :transparent,
                        aspect=DataAspect(), 
                        xticksvisible = false, 
                        xticklabelsvisible = false, 
                        yticksvisible = false, 
                        yticklabelsvisible = false,
                        leftspinevisible = false,
                        rightspinevisible = false,
                        topspinevisible = false,
                        bottomspinevisible = false,
                        xautolimitmargin = (0.0,0.0),
                        yautolimitmargin = (0.0,0.0)
            )
    hm = heatmap!(ax, first_axis, second_axis, pic; colormap = :oslo)
    save(filename, ax.scene, px_per_unit = 2)
end

filename = "original/image.png"
_write_to_png(first_image', filename)

for i in 1:16 
    p_small = processed_image₁[i];
    file_name = "split/"*string(i)*".png"
    _write_to_png(p_small', file_name)
end

for i in 1:16
    p_small = processed_image₂[i]
    file_name = "flatten/"*string(i)*".png"
    _write_to_png(p_small', file_name)
end

p_final = fully_processed_image
_write_to_png(p_final', "final/image.png")