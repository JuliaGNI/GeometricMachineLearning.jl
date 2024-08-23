using Test
using GeometricMachineLearning
using GeometricMachineLearning: split_and_flatten
using GeometricMachineLearning: patch_index
using GeometricMachineLearning: within_patch_index
using GeometricMachineLearning: index_conversion
import Zygote, Random 

Random.seed!(1234)

"""
This function tests is used to test if all the patch nubmers are assigned correctly with `index_conversion`, i.e. tests `patch_index` by inverting it.
"""
function reverse_index(i::Integer, j::Integer, patch_length=7)
    opt_i = i%patch_length==0 ? 1 : 0
    within_patch_index = i%patch_length + opt_i*patch_length, (i÷patch_length - opt_i + 1)

    sqrt_number_patches = 28÷patch_length
    opt_j = j%sqrt_number_patches==0 ? 1 : 0 
    patch_index = j%sqrt_number_patches + opt_j*sqrt_number_patches, (j÷sqrt_number_patches - opt_j + 1)
    (patch_index[1]-1)*patch_length + within_patch_index[1], (patch_index[2]-1)*patch_length + within_patch_index[2]
end 

"""
This function uses `reverse_index` to test `index_conversion`, i.e. checks if the functions are invertible. 
"""
function test_index_conversion(patch_lengths=(2, 4, 7, 14)) 
    for patch_length in patch_lengths
        number_of_patches = (28÷patch_length)^2
        for i in 1:28 
            for j in 1:28
                @test reverse_index(index_conversion(i, j, patch_length, number_of_patches)..., patch_length) == (i, j)
            end
        end
    end
end

"""
This function tests if `onehotbatch` does what it should; i.e. convert a vector of integers to a one-hot-tensor.
"""
function test_onehotbatch(V::AbstractVector{T}) where {T<:Integer} 
    V_encoded = onehotbatch(V)
    for (i, v) in zip(length(V), V)
        @test sum(V_encoded[:,1,i]) == 1
        @test V_encoded[v, 1, i] == 1
    end
end

test_onehotbatch([1, 2, 5, 0])

@doc raw"""
Generates an MNIST-like dummy data set.
"""
function generate_dummy_mnist(dim₁=28, dim₂=28, number_images=100, T=Float32)

    train_x = rand(T, dim₁, dim₂, number_images)
    train_y = Int.(ceil.(10 * rand(T, number_images))) .- 1
    train_x, train_y
end

function test_optimizer_for_classification_layer(; dim₁=28, dim₂=28, number_images=100, patch_length=7, T=Float32)
    dl = DataLoader(generate_dummy_mnist(dim₁, dim₂, number_images, T)...; patch_length=patch_length)

    activation_function(x) = tanh.(x)
    model = ClassificationLayer(patch_length * patch_length, 10, activation_function)

    ps = initialparameters(model, CPU(), T)   
    loss = FeedForwardLoss()
    loss_dl(model::GeometricMachineLearning.AbstractExplicitLayer, ps::Union{Tuple, NamedTuple}, dl::DataLoader) = loss(model, ps, dl.input, dl.output)
    loss₁ = loss_dl(model, ps, dl)

    opt = Optimizer(GradientOptimizer(), ps)
    dx = Zygote.gradient(ps -> loss_dl(model, ps, dl), ps)[1]
    λY = GlobalSection(ps)
    optimization_step!(opt, λY, ps, dx)
    loss₂ = loss_dl(model, ps, dl)

    @test loss₂ < loss₁
end

test_index_conversion()
test_optimizer_for_classification_layer()