# Used by the training scripts that take the parameters of the system. These helpers turn an
# ensemble of trajectories into the data and the system parameters that `ParametricDataLoader`
# takes: one `NamedTuple` of system parameters per sample.
#
# They are here rather than in each script because the reshaping is the same for every script of
# this kind. They do not belong in `scripts/` either -- generating and reshaping data sets is
# `GMLDatasets`' job, https://github.com/JuliaGNI/GMLDatasets.jl/issues/5 tracks moving them there --
# so treat this file as a staging post.

using GeometricMachineLearning: ParametricDataLoader

"""
    turn_parameters_into_correct_format(t, IC)

The vector of `NamedTuple`s to be used by [`ParametricDataLoader`](@ref): one `(t = …,)` per time
step of the trajectory, repeated once per initial condition of `IC`.

There is one entry fewer than there are time steps in `t`, because a sample is a pair of consecutive
states and the last one is only ever a target.
"""
function turn_parameters_into_correct_format(t::AbstractVector, IC::AbstractVector{<:NamedTuple})
    vec_of_params = NamedTuple[]
    for time_step in t
        time_step == t[end] || push!(vec_of_params, (t = time_step,))
    end
    vcat((vec_of_params for _ in axes(IC, 1))...)
end

@doc raw"""
    turn_q_p_data_into_correct_format(qp)

Turn a `NamedTuple` of ``(q, p)`` data into the two tensors of the correct format.

This is the tricky part, as the structure of the arrays has to conform to the structure of the
parameters. A [`ParametricDataLoader`](@ref) reads a ``(\mathrm{dim}, nt, m)`` tensor as `m`
trajectories of `nt` time steps and takes *one* parameter for each of the `m` trajectories, so a
parameter that depends on the time -- as the time does here -- has to make every sample a trajectory
of its own: a pair of consecutive states.

`qp.q` and `qp.p` are the ``(m, nt)`` matrices of `m` initial conditions, one time series each, so
both become ``(1, 2, m \cdot (nt - 1))`` tensors: axis 2 is the start state and the state one time
step later, and the samples are ordered by initial condition and then by time, which is the order
`turn_parameters_into_correct_format` puts the parameters in. `turned.q[1, 1, k]` is the state at
the start of sample `k` and `turned.q[1, 2, k]` the state one time step later.
"""
function turn_q_p_data_into_correct_format(
        qp::NamedTuple{(:q, :p), <:Tuple{AbstractMatrix{T}, AbstractMatrix{T}}}) where {T}
    m, n_steps = size(qp.q)
    n_samples = n_steps - 1
    q_array = zeros(T, 1, 2, n_samples * m)
    p_array = zeros(T, 1, 2, n_samples * m)
    for j in 1:m, i in 1:n_samples

        sample = (j - 1) * n_samples + i
        q_array[1, 1, sample] = qp.q[j, i]
        q_array[1, 2, sample] = qp.q[j, i + 1]
        p_array[1, 1, sample] = qp.p[j, i]
        p_array[1, 2, sample] = qp.p[j, i + 1]
    end
    (q = q_array, p = p_array)
end

"""
    load_time_dependent_harmonic_oscillator_with_parametric_data_loader(qp, t, IC)

The [`ParametricDataLoader`](@ref) of a harmonic oscillator whose parameters depend on the time,
with one `(t = …,)` per sample: `qp` is the `(q, p)` data of the trajectories in `IC`, and `t` their
time grid.
"""
function load_time_dependent_harmonic_oscillator_with_parametric_data_loader(
        qp::NamedTuple{(:q, :p)}, t::AbstractVector, IC::AbstractVector{<:NamedTuple})
    qp_reformatted = turn_q_p_data_into_correct_format(qp)
    t_reformatted = turn_parameters_into_correct_format(t, IC)
    ParametricDataLoader(qp_reformatted, t_reformatted)
end

@doc raw"""
    forced_harmonic_oscillator_solution(t, IC; omega, Omega, F)

The analytic solution of the sinusoidally forced harmonic oscillator, as ``(q, p)`` arrays of size
``(m, \mathtt{length}(t))`` for the ``m`` initial conditions of `IC`.

The oscillator is ``\ddot q = -\omega^2 q + F \sin(\Omega t)``, whose solution carries the amplitude
``F/(\omega^2 - \Omega^2)``; `Omega` must therefore not be `omega`.
"""
function forced_harmonic_oscillator_solution(
        t::AbstractVector{T}, IC::AbstractVector{<:NamedTuple};
        omega::Real, Omega::Real, F::Real) where {T <: AbstractFloat}
    m = length(IC)
    q = zeros(T, m, length(t))
    p = zeros(T, m, length(t))
    amplitude = F / (omega^2 - Omega^2)
    for (i, time) in enumerate(t), (j, initial_condition) in enumerate(IC)

        q[j, i] = (initial_condition.p - Omega * amplitude) / omega * sin(omega * time) +
                  initial_condition.q * cos(omega * time) + amplitude * sin(Omega * time)
        p[j, i] = -omega^2 * initial_condition.q * sin(omega * time) +
                  (initial_condition.p - Omega * amplitude) * cos(omega * time) +
                  Omega * amplitude * cos(Omega * time)
    end
    (q = q, p = p)
end
