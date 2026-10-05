using Test
using GeometricMachineLearning
using LinearAlgebra: I
using GeometricMachineLearning: global_section
# `Ω` is GeometricOptimizers', along with the manifolds themselves, and this package neither adds a
# method to it nor re-exports it. It was reached here as `GeometricMachineLearning.Ω`, which worked
# only because the module file carried an `import` that nothing in `src/` used.
using GeometricOptimizers: Ω

@testset "Manifold docstring examples" begin
    Y = StiefelManifold([1 0; 0 1; 0 0; 0 0])
    Δ = [1 2; 3 4; 5 6; 7 8]
    @test rgrad(Y, Δ) == [0 -1; 1 0; 5 6; 7 8]

    Y = GrassmannManifold([1 0; 0 1; 0 0; 0 0])
    @test rgrad(Y, Δ) == [0 0; 0 0; 5 6; 7 8]

    E = GrassmannManifold(StiefelProjection(5, 2))
    Δ = [0.0 0.0; 0.0 0.0; 2.0 3.0; 4.0 5.0; 6.0 7.0]
    ΩE = Matrix(Ω(E, Δ))
    @test ΩE ≈ -ΩE'
    @test ΩE[3:5, 1:2] ≈ Δ[3:5, :]

    E = StiefelManifold(StiefelProjection(5, 2))
    Δ = [0.0 -1.0; 1.0 0.0; 2.0 3.0; 4.0 5.0; 6.0 7.0]
    ΩE = Matrix(Ω(E, Δ))
    expected = [0.0 -1.0 -2.0 -4.0 -6.0; 1.0 0.0 -3.0 -5.0 -7.0;
                2.0 3.0 0.0 0.0 0.0; 4.0 5.0 0.0 0.0 0.0; 6.0 7.0 0.0 0.0 0.0]
    @test ΩE ≈ expected

    Y = StiefelManifold([1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0])
    section = Matrix(global_section(Y))
    @test size(section) == (4, 2)
    @test Y' * section ≈ zeros(2, 2) atol = 1e-12
    @test section' * section ≈ I(2)
end
