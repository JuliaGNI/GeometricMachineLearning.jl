using SafeTestsets

@safetestset "PSD tests" begin
    include("psd_architecture_tests.jl")
end
@safetestset "SymplecticAutoencoder tests" begin
    include("symplectic_autoencoder_tests.jl")
end
@safetestset "Check if autoencoder error is lower than PSD error" begin
    include("sae_error_lower_than_psd_error.jl")
end
@safetestset "Check reduced model" begin
    include("reduced_system.jl")
end
