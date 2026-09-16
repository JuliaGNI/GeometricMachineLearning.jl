using SafeTestsets

@safetestset "_custom_mul: NetworkParameters gradient structure" begin
    include("double_multiplication_network_parameters_gradient.jl")
end
@safetestset "Symplectic attention: NetworkParameters gradient structure" begin
    include("symplectic_attention_network_parameters_gradient.jl")
end
@safetestset "map_to_cpu" begin
    include("map_to_cpu_tests.jl")
end
@safetestset "changebackend" begin
    include("changebackend_tests.jl")
end
@safetestset "HDF5 save/load for GML special array types" begin
    include("hdf5_support.jl")
end
