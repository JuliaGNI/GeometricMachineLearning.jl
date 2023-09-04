using Test
using GeometricMachineLearning
using GeometricMachineLearning: split_and_flatten
using GeometricMachineLearning: patch_index
using MLDatasets
import Zygote

"""
This function tests if all the patch nubmers are assigned correctly, i.e. tests patch_index.
"""
function test_correct_assignment(patch_length=7, number_of_patches=16)
    count = zeros(16)
    for i in 1:(Int(√number_of_patches)*patch_length)
        for j in 1:(Int(√number_of_patches)*patch_length)
            patch_index₁ =  patch_index(i,j,patch_length,number_of_patches)
            count[patch_index₁] += 1
        end 
    end

    for count_element in count
        @test count_element == 49
    end
end

test_correct_assignment()

function test_onehotbatch(V::AbstractVector{T}) where {T<:Integer} 
    V_encoded = onehotbatch(V)
    for i in length(V)
        @test sum(V_encoded[:,i]) == 1
    end
end

test_onehotbatch([1, 2, 5, 0])

function test_within_patch_index()
    within_patch_indices = zeros(49)
    for i in 1:28
        for j in 1:28 
            within_patch_indices[within_patch_index(i,j,7)] += 1
        end
    end
    for within_patch_number in within_batch_indices 
        @test within_patch_number == 16
    end
end

####### MNIST data set 

train_x, train_y = MLDatasets.MNIST(split=:train)[:]

dl = DataLoader(train_x, train_y)
redraw_batch(dl)

model = Dense(49, 10, tanh)
ps = initialparameters(CPU(), Float32, model)
loss₁ = loss(model, ps, dl)

opt = Optimizer(GradientOptimizer(), ps)
dx = Zygote.gradient(ps -> loss(model, ps, dl), ps)[1]
optimization_step!(opt, model, ps, dx)
loss₂ = loss(model, ps, dl)

@test loss₂ < loss₁