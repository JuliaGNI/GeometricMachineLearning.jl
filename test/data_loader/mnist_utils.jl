using Test
using GeometricMachineLearning
using GeometricMachineLearning: split_and_flatten
using GeometricMachineLearning: patch_index
using MLDatasets

"""
This function tests if all the patch nubmers are assigned correctly.
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

train_x, train_y = MLDatasets.MNIST(split=:train)[:]

dl = DataLoader(train_x, train_y)
redraw_batch(dl)