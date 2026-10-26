# The constructors of `src/data_loader/data_loader.jl`: the `autoencoder` default and the keyword
# set of every method, the fifth type parameter of the struct (#211), the copy loops out of a
# solution and an ensemble, the element-type conversion of `DataLoader(dl, T)`, and the
# `(input, output)` method that lets a classifier's data be built without naming the parametric type.

using GeometricMachineLearning
using GeometricMachineLearning: solve!
using Test
import Random

using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.HarmonicOscillator: hodeensemble, hodeproblem
using GeometricProblems.RigidBody: odeensemble, odeproblem, default_parameters

include("../helpers/eltypes.jl")

Random.seed!(1234)

# The kind of a loader is its fourth type parameter.
kind(dl::DataLoader) = typeof(dl).parameters[4]

@testset "every constructor defaults autoencoder = false ($T)" for T in REAL_ELTYPES
    matrix = rand(T, 3, 10)
    tensor = rand(T, 3, 10, 2)
    @test kind(DataLoader(matrix; suppress_info = true)) === :TimeSeries
    @test kind(DataLoader(rand(T, 10); suppress_info = true)) === :TimeSeries
    @test kind(DataLoader(tensor; suppress_info = true)) === :TimeSeries
    @test kind(DataLoader((q = matrix, p = matrix); suppress_info = true)) === :TimeSeries
    # A literal keyword fixes the loader's kind, so the tensor constructors infer a concrete type.
    @test (@inferred (() -> DataLoader(tensor; suppress_info = true))()) isa DataLoader
    @test (@inferred (() -> DataLoader((q = tensor, p = tensor);
        suppress_info = true))()) isa DataLoader
    @test (@inferred (() -> DataLoader(tensor; autoencoder = true,
        suppress_info = true))()) isa DataLoader
    @test (@inferred (() -> DataLoader((q = tensor, p = tensor); autoencoder = true,
        suppress_info = true))()) isa DataLoader
    # ... and so do the matrix and vector constructors, which forward to them
    vector = rand(T, 10)
    @test (@inferred (() -> DataLoader(matrix; suppress_info = true))()) isa DataLoader
    @test (@inferred (() -> DataLoader(vector; autoencoder = true,
        suppress_info = true))()) isa DataLoader
    @test (@inferred (() -> DataLoader((q = matrix, p = matrix);
        suppress_info = true))()) isa DataLoader
    @test (@inferred (() -> DataLoader((q = vector, p = vector); autoencoder = true,
        suppress_info = true))()) isa DataLoader
    @test kind(DataLoader((q = tensor, p = tensor); suppress_info = true)) === :TimeSeries
    @test kind(DataLoader((q = rand(T, 10), p = rand(T, 10)); suppress_info = true)) ===
          :TimeSeries
    # ... and `autoencoder = true` is what makes a matrix a set of samples
    @test kind(DataLoader(matrix; autoencoder = true, suppress_info = true)) ===
          :RegularData
    @test kind(DataLoader((q = rand(T, 10), p = rand(T, 10));
        autoencoder = true, suppress_info = true)) === :RegularData
    # a loader with an output is a set of samples, whatever the two element types are
    @test kind(DataLoader(matrix, matrix; suppress_info = true)) === :RegularData
end

# The matrix message names the second axis by the kind actually built.
@testset "the matrix message names the axis by the kind" begin
    @test_logs (:info, r"time steps") DataLoader(rand(Float32, 3, 10))
    @test_logs (:info, r"number of parameters") DataLoader(
        rand(Float32, 3, 10); autoencoder = true)
end

# `suppress_info` is a keyword in every method of a data set, a solution and an ensemble.
@testset "suppress_info is a keyword everywhere" begin
    solution = integrate(
        hodeproblem(; timespan = (0.0, 1.0), timestep = 0.25), ImplicitMidpoint())
    @test DataLoader(solution; suppress_info = true) isa DataLoader
    @test_throws MethodError DataLoader(solution, true)

    ensemble = integrate(
        odeensemble([(q = [0.0, 0.2, 1.0],), (q = [0.1, 0.3, 1.0],)];
            timespan = (0.0, 0.4), timestep = 0.2, parameters = default_parameters()),
        ImplicitMidpoint())
    @test DataLoader(ensemble; suppress_info = true) isa DataLoader
    @test_throws MethodError DataLoader(ensemble, true)
end

# The copy loops of `data_tensors_from_geometric_solution` build one 1-based tensor per coordinate.
# The rigid-body `odeensemble` is the case whose `dataser` carries `(t, q, q̇)` and no `p`, which is
# the case its own method is for.
@testset "a solution and an ensemble become 1-based tensors" begin
    ics = [(q = [sin(v), 0.0, cos(v)],) for v in 0.1:0.5:2.0]
    ensemble = integrate(
        odeensemble(ics; timespan = (0.0, 1.0), timestep = 0.2,
            parameters = default_parameters()), ImplicitMidpoint())
    dl = DataLoader(ensemble; suppress_info = true)
    @test dl.input isa Array{Float64, 3}
    @test kind(dl) === :TimeSeries
    @test size(dl.input) == (3, length(ensemble.t), length(ensemble.s))
    for (i, solution) in pairs(ensemble.s), t in 0:(length(solution.t) - 1)

        @test dl.input[:, t + 1, i] == solution.q[t]
    end

    # one solution of the same problem
    solution = integrate(
        odeproblem(; timespan = (0.0, 1.0), timestep = 0.2), ImplicitMidpoint())
    dl_one = DataLoader(solution; suppress_info = true)
    @test dl_one.input isa Array{Float64, 3}
    @test size(dl_one.input) == (3, length(solution.t), 1)
    for t in 0:(length(solution.t) - 1)
        @test dl_one.input[:, t + 1, 1] == solution.q[t]
    end

    # and a `(q, p)` ensemble: the two coordinates are not swapped
    ensemble_qp = integrate(
        hodeensemble(; timespan = (0.0, 0.5), timestep = 0.25), ImplicitMidpoint())
    dl_qp = DataLoader(ensemble_qp; suppress_info = true)
    @test dl_qp.input isa NamedTuple{(:q, :p)}
    @test size(dl_qp.input.q) == (1, length(ensemble_qp.t), length(ensemble_qp.s))
    for (i, sol) in pairs(ensemble_qp.s), t in 0:(length(sol.t) - 1)

        @test dl_qp.input.q[:, t + 1, i] == sol.q[t]
        @test dl_qp.input.p[:, t + 1, i] == sol.p[t]
    end

    # and one `(q, p)` solution: the two coordinates are not swapped
    solution_qp = integrate(
        hodeproblem(; timespan = (0.0, 1.0), timestep = 0.25), ImplicitMidpoint())
    dl_qp_one = DataLoader(solution_qp; suppress_info = true)
    @test dl_qp_one.input isa NamedTuple{(:q, :p)}
    @test size(dl_qp_one.input.q) == (1, length(solution_qp.t), 1)
    for t in 0:(length(solution_qp.t) - 1)
        @test dl_qp_one.input.q[:, t + 1, 1] == solution_qp.q[t]
        @test dl_qp_one.input.p[:, t + 1, 1] == solution_qp.p[t]
    end
end

# The reproduction script `scripts/reproduction/symplectic_autoencoders/training.jl` has a `solve!`
# of a PSD architecture on a snapshot matrix, and `solve!` takes a `:RegularData` loader. That is
# why its `DataLoader(data)` passes `autoencoder = true`, and why a matrix that does not is a
# `MethodError` rather than a training run on input-shifted targets.
@testset "solve! of a PSD architecture needs a samples loader" begin
    data = rand(Float32, 6, 20)
    nn = NeuralNetwork(PSDArch(6, 2), CPU(), Float32)
    @test isfinite(solve!(nn, DataLoader(data; autoencoder = true, suppress_info = true)))
    @test_throws MethodError solve!(nn, DataLoader(data; suppress_info = true))
end

# #211: the two fields that are `nothing` for an unsupervised loader are concrete.
@testset "#211: the output fields are concrete in every loader" begin
    loaders = (DataLoader(rand(Float32, 2, 3, 4); suppress_info = true),
        DataLoader(rand(Float32, 2, 3, 4), rand(Float32, 2, 3, 4); suppress_info = true),
        DataLoader(rand(Float32, 2, 3, 4); autoencoder = true, suppress_info = true),
        DataLoader(rand(Float32, 2, 3); autoencoder = true, suppress_info = true))
    for dl in loaders
        @test isconcretetype(fieldtype(typeof(dl), :output_dim))
        @test isconcretetype(fieldtype(typeof(dl), :output_time_steps))
        @test (fieldtype(typeof(dl), :output_dim) === Nothing) == (dl.output === nothing)
    end
end

# The four-parameter constructor fills the fifth type parameter in, and it takes the counts as any
# integer type: `GMLDatasets` builds a loader by naming the first four and handing it the `size`s.
@testset "the four-parameter constructor converts its counts" begin
    input = rand(Float32, 2, 3, 4)
    output = zeros(Int, 3, 1, 4)
    dl = DataLoader{Float32, typeof(input), typeof(output), :TimeSeries}(input, output,
        Int32(2), Int32(3), Int32(4), Int32(3), Int32(1))
    @test dl isa DataLoader{Float32, Array{Float32, 3}, Array{Int, 3}, :TimeSeries, Int}
    @test (dl.input_dim, dl.input_time_steps, dl.n_params) == (2, 3, 4)
    @test (dl.output_dim, dl.output_time_steps) == (3, 1)
    @test dl.output === output

    no_output = DataLoader{Float32, typeof(input), Nothing, :RegularData}(input, nothing,
        Int8(2), Int8(3), Int8(4), nothing, nothing)
    @test no_output isa
          DataLoader{Float32, Array{Float32, 3}, Nothing, :RegularData, Nothing}
    @test (no_output.output_dim, no_output.output_time_steps) == (nothing, nothing)
end

# The conversion to another element type keeps the kind and the counts, and the other backend
# branch is the identity.
@testset "DataLoader(dl, T) converts and keeps the kind ($T)" for T in REAL_ELTYPES
    other = T === Float32 ? Float64 : Float32
    dl = DataLoader(rand(other, 2, 3, 4); suppress_info = true)
    converted = DataLoader(dl, T)
    @test eltype(converted) === T
    @test converted.input isa Array{T, 3}
    @test converted.input == T.(dl.input)
    @test kind(converted) === kind(dl)
    @test (converted.input_dim, converted.input_time_steps, converted.n_params) ==
          (dl.input_dim, dl.input_time_steps, dl.n_params)
    @test converted.output === nothing
    # the same backend and the same element type is the same tensor
    @test DataLoader(dl, other).input === dl.input
end

# A classifier's real input and integer one-hot output, without naming the parametric type.
@testset "a loader of an input and an integer output" begin
    input = rand(Float32, 4, 3, 6)
    output = zeros(Int64, 3, 1, 6)
    dl = DataLoader(input, output; suppress_info = true)
    @test kind(dl) === :RegularData
    @test (dl.input_dim, dl.input_time_steps, dl.n_params) == (4, 3, 6)
    @test (dl.output_dim, dl.output_time_steps) == (3, 1)
    @test eltype(dl) === Float32
    @test dl.output === output
    # the same element type in both arguments still reaches a method, and so does an `Int`/`Int` pair
    @test DataLoader(input, rand(Float32, 3, 1, 6); suppress_info = true) isa DataLoader
    @test DataLoader(zeros(Int, 2, 3, 4), zeros(Int64, 2, 3, 4); suppress_info = true) isa
          DataLoader{Int, Array{Int, 3}, Array{Int64, 3}, :RegularData}
    # two element types of which neither is an integer stay a `MethodError`
    @test_throws MethodError DataLoader(
        rand(Float32, 2, 3, 4), rand(Float64, 2, 3, 4); suppress_info = true)
    # the third axis is still the parameter index of both
    @test_throws AssertionError DataLoader(
        rand(Float32, 2, 3, 4), zeros(Int, 3, 1, 5); suppress_info = true)
end
