using Test
using GeometricMachineLearning
using GeometricMachineLearning: split_and_flatten
using GeometricMachineLearning: patch_index
using GeometricMachineLearning: within_patch_index
using GeometricMachineLearning: index_conversion
using MLDatasets
import Zygote

"""
This function tests if all the patch nubmers are assigned correctly, i.e. tests patch_index.
"""
### Test if mapping is invertible
function reverse_index(i::Integer, j::Integer, patch_length=7)
    opt_i = i%patch_length==0 ? 1 : 0
    within_patch_index = i%patch_length + opt_i*patch_length, (i÷patch_length - opt_i + 1)

    sqrt_number_patches = 28÷patch_length
    opt_j = j%sqrt_number_patches==0 ? 1 : 0 
    patch_index = j%sqrt_number_patches + opt_j*sqrt_number_patches, (j÷sqrt_number_patches - opt_j + 1)
    (patch_index[1]-1)*patch_length + within_patch_index[1], (patch_index[2]-1)*patch_length + within_patch_index[2]
end 

# test if this is the inverse of the other batch index conversion!
patch_lengths = (2, 4, 7, 14)
for patch_length in patch_lengths
    number_of_patches = (28÷patch_length)^2
      for i in 1:28 
        for j in 1:28
            @test reverse_index(index_conversion(i, j, patch_length, number_of_patches)..., patch_length) == (i, j)
        end
    end
end

function test_onehotbatch(V::AbstractVector{T}) where {T<:Integer} 
    V_encoded = onehotbatch(V)
    for i in length(V)
        @test sum(V_encoded[:,i]) == 1
    end
end

test_onehotbatch([1, 2, 5, 0])
####### MNIST data set 

train_x, train_y = MLDatasets.MNIST(split=:train)[:]

dl = DataLoader(train_x, train_y)
redraw_batch!(dl)

model = Dense(49, 10, tanh)
ps = initialparameters(CPU(), Float32, model)
loss₁ = loss(model, ps, dl)

opt = Optimizer(GradientOptimizer(), ps)
dx = Zygote.gradient(ps -> loss(model, ps, dl), ps)[1]
optimization_step!(opt, model, ps, dx)
loss₂ = loss(model, ps, dl)

@test loss₂ < loss₁
