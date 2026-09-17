using SafeTestsets

@safetestset "Check parameterlength" begin
    include("check_parameterlengths.jl")
end
@safetestset "Hamiltonian Neural Network" begin
    include("hamiltonian_neural_network_tests.jl")
end
@safetestset "Lagrangian Neural Network" begin
    include("lagrangian_neural_network_tests.jl")
end
@safetestset "SympNet integrator" begin
    include("sympnet_integrator.jl")
end
