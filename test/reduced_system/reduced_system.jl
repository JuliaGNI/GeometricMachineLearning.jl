using GeometricMachineLearning
using GeometricProblems.CoupledHarmonicOscillator: hodeproblem, hodeensemble
using GeometricIntegrators: integrate, ImplicitMidpoint, ExplicitEulerRK, ExplicitMidpoint
using FiniteDifferences: jacobian, central_fdm
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(123)

function set_up_reduced_systems(reduced_dim::Integer, integrator, ::Type{T}) where {T}
    pr = hodeproblem()

    sol = integrate(pr, integrator)

    dl = DataLoader(sol)

    model1 = PSDArch(dl.input_dim, reduced_dim)

    # Here the number of decoder blocks is set manually because the default is too big! 
    model2 = SymplecticAutoencoder(
        dl.input_dim, reduced_dim; activation = x -> log(one(x) + exp(x)),
        n_encoder_layers = 20, n_decoder_layers = 10, n_decoder_blocks = 2)

    nn1 = NeuralNetwork(model1, T)

    nn2 = NeuralNetwork(model2, T)

    rs1 = HRedSys(hodeproblem(), encoder(nn1), decoder(nn1); integrator = integrator)

    rs2 = HRedSys(hodeproblem(), encoder(nn2), decoder(nn2); integrator = integrator)

    rs1, rs2, nn1, nn2
end

# We expect the neural network projection error to be higher than the PSD projection error if we
# don't train.
#
# The reduction errors are asserted finite and not ordered. `projection_error` compares the two
# autoencoders on the *full* solution, but `reduction_error` integrates the *reduced* system, and the
# reduced vector field of an untrained `SymplecticAutoencoder` is one an explicit method diverges on:
# `reduction_error(rs2)` can come back `NaN`, which is not ordered against anything. An implicit
# method keeps both errors numbers, which is all an untrained network supports.
function test_reduced_vector_fields(reduced_dim::Integer, integrator, ::Type{T}) where {T}
    rs1, rs2, nn1, _ = set_up_reduced_systems(reduced_dim, integrator, T)

    # the networks the reduced systems are built from are of the requested element type, and so is
    # what their decoder returns
    @test eltype(decoder(nn1)(rand(T, reduced_dim))) === T
    @test projection_error(rs1) < projection_error(rs2)
    if integrator isa ImplicitMidpoint
        @test isfinite(reduction_error(rs1)) && isfinite(reduction_error(rs2))
    end
end

function test_if_reduced_vector_fields_are_divergence_free(
        v_reduced::Function, f_reduced::Function, parameters::NamedTuple; tol = 1e-10)
    function v_reduced_explicit(q, p)
        v = zero(q)
        v_reduced(v, 0, q, p, parameters)
        v
    end
    function f_reduced_explicit(q, p)
        f = zero(p)
        f_reduced(f, 0, q, p, parameters)
        f
    end

    q̃ = rand(1)
    p̃ = rand(1)
    compute_derivative_q(f) = sum(jacobian(central_fdm(10, 1), q -> f(q, p̃), q̃)[1])
    compute_derivative_p(f) = sum(jacobian(central_fdm(10, 1), p -> f(q̃, p), p̃)[1])
    div_estimate = compute_derivative_q(v_reduced_explicit) +
                   compute_derivative_p(f_reduced_explicit)
    @test abs(div_estimate) < tol
end

function test_if_reduced_vector_fields_are_divergence_free(rs::HRedSys)
    test_if_reduced_vector_fields_are_divergence_free(rs.v_reduced, rs.f_reduced, rs.parameters)

    nothing
end

# The divergence of the reduced fields is evaluated in the problem's own `Float64`: the state and the
# vector fields are the problem's. The bound below is `Float64`'s, and the three calls name
# `Float64` rather than looping over `REAL_ELTYPES`.
function check_if_reduced_vector_fields_are_divergence_free(
        reduced_dim::Integer, integrator,
        ::Type{T}) where {T}
    rs1, rs2, _, _ = set_up_reduced_systems(reduced_dim, integrator, T)

    test_if_reduced_vector_fields_are_divergence_free(rs1)
    test_if_reduced_vector_fields_are_divergence_free(rs2)

    nothing
end

for T in REAL_ELTYPES
    test_reduced_vector_fields(2, ImplicitMidpoint(), T)
    test_reduced_vector_fields(2, ExplicitMidpoint(), T)
end

check_if_reduced_vector_fields_are_divergence_free(2, ImplicitMidpoint(), Float64)
check_if_reduced_vector_fields_are_divergence_free(2, ExplicitMidpoint(), Float64)
check_if_reduced_vector_fields_are_divergence_free(2, ExplicitEulerRK(), Float64)

@doc raw"""
`HRedSys.timespan` carries its own type parameter and follows the constructor's argument, the way
the neighbouring `timestep::T` field does. The standard constructor takes `timespan::Tuple` and
forwards it unchanged, so a field declared `Tuple{Int, Int}` would silently truncate a conventional
`(0.0, 1.0)` to `(0, 1)` and throw an `InexactError` on `(0.0, 1.5)`.
"""
function test_hredsys_timespan_is_not_truncated()
    model = PSDArch(4, 2)
    nn = NeuralNetwork(model)
    v_full(v, t, q, p, parameters) = (v .= 0)
    f_full(f, t, q, p, parameters) = (f .= 0)
    h_full(t, q, p, parameters) = 0.0
    ics = (q = GeometricMachineLearning.StateVariable(zeros(4)),
        p = GeometricMachineLearning.StateVariable(zeros(4)))

    rs = HRedSys(
        4, 2, encoder(nn), decoder(nn), v_full, f_full, h_full, (0.0, 1.0), 0.1, ics;
        parameters = NamedTuple(), integrator = ImplicitMidpoint())
    @test rs.timespan === (0.0, 1.0)

    # A non-integer endpoint must not throw an `InexactError`.
    rs2 = HRedSys(
        4, 2, encoder(nn), decoder(nn), v_full, f_full, h_full, (0.0, 1.5), 0.1, ics;
        parameters = NamedTuple(), integrator = ImplicitMidpoint())
    @test rs2.timespan === (0.0, 1.5)
end

test_hredsys_timespan_is_not_truncated()

@doc raw"""
`HRedSys` takes its integrator from the caller and its parameters from a constructor of its own:
the positional constructor defaults `parameters` to `NullParameters()` and requires `integrator`,
and the `HODEProblem` and `HODEEnsemble` constructors require it too. None of the three names an
integrator, so a call that forgets one raises an `UndefKeywordError` instead of building a system
around a scheme the caller never chose.
"""
function test_hredsys_keywords()
    model = PSDArch(4, 2)
    nn = NeuralNetwork(model)
    v_full(v, t, q, p, parameters) = (v .= 0)
    f_full(f, t, q, p, parameters) = (f .= 0)
    h_full(t, q, p, parameters) = 0.0
    ics = (q = GeometricMachineLearning.StateVariable(zeros(4)),
        p = GeometricMachineLearning.StateVariable(zeros(4)))

    @test_throws UndefKeywordError HRedSys(
        4, 2, encoder(nn), decoder(nn), v_full, f_full, h_full, (0.0, 1.0), 0.1, ics)
    @test_throws UndefKeywordError HRedSys(
        4, 2, encoder(nn), decoder(nn), v_full, f_full, h_full, (0.0, 1.0), 0.1, ics;
        parameters = NamedTuple())
    # The two constructors that take an equation require it as well.
    @test_throws UndefKeywordError HRedSys(hodeproblem(), encoder(nn), decoder(nn))
    @test_throws UndefKeywordError HRedSys(hodeensemble(), encoder(nn), decoder(nn))

    # A call that names the integrator and no parameters gets the empty parameter set that the
    # vector fields above are written against.
    rs = HRedSys(
        4, 2, encoder(nn), decoder(nn), v_full, f_full, h_full, (0.0, 1.0), 0.1, ics;
        integrator = ImplicitMidpoint())
    @test rs.parameters isa GeometricMachineLearning.GeometricBase.NullParameters
    @test rs.integrator === ImplicitMidpoint()
end

test_hredsys_keywords()
