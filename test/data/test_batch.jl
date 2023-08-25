using GeometricMachineLearning 
using Test

include("data_generation.jl")
include("../macro_testerror.jl")


#########################################
# Test complete_batch_size
#########################################

training_data = tra_ps_data
sampled_data = sam_ps_data
ti = BasicSympNet()

@test complete_batch_size(training_data, ti, (2,8,2))   == (2,8,2)
@test complete_batch_size(training_data, ti, (2,8))     == (2,8,2)
@test complete_batch_size(training_data, ti, 8)         == (2,8,2)
@test complete_batch_size(training_data, ti, missing)   == (1, 2, 2)
@test complete_batch_size(sampled_data, ti, 5)          == 5
@test complete_batch_size(sampled_data, ti, missing)    == 6

#########################################
# Test get_batch function
#########################################

index_batch = get_batch(training_data, (2,2,2))
@test length(index_batch) == 2
for i in index_batch
    x,y = i 
    @test 1<= x <= get_nb_trajectory(training_data)
    @test 1<= y <= get_length_trajectory(training_data, x)
end

#########################################
# Test check_batch_size
#########################################




