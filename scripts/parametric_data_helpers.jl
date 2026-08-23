# Shared by the parametric/forced training scripts in this directory. These helpers reshape a
# trajectory ensemble into the `(system dimension, time, parameter)` layout `ParametricDataLoader`
# wants, with one `NamedTuple` of system parameters per column.
#
# They are here rather than in each script because all three used to carry a verbatim copy. They do
# not really belong in `scripts/` either -- generating and reshaping data sets is `GMLDatasets`'
# job -- https://github.com/JuliaGNI/GMLDatasets.jl/issues/5 tracks moving them there -- so treat
# this file as a staging post.

using GeometricMachineLearning
using GeometricMachineLearning: QPT, QPT2

"""
Turn a vector of numbers into a vector of `NamedTuple`s to be used by `ParametricDataLoader`.
"""
function turn_parameters_into_correct_format(t::AbstractVector, IC::AbstractVector{<:NamedTuple})
	vec_of_params = NamedTuple[]
	for time_step ∈ t
		time_step == t[end] || push!(vec_of_params, (t = time_step, ))
	end
	vcat((vec_of_params for _ in axes(IC, 1))...)
end

@doc raw"""
Turn a `NamedTuple` of ``(q,p)`` data into two tensors of the correct format.

This is the tricky part as the structure of the input array(s) needs to conform with the structure of the parameters.

Here the data are rearranged in an array of size ``(n, 2, t_f - 1)`` where ``[t_0, t_1, \ldots, t_f]`` is the vector storing the time steps.

If we deal with different initial conditions as well, we still put everything into the third (parameter) axis.

# Example

```jldoctest
using GeometricMachineLearning

q = [1. 2. 3.; 4. 5. 6.]
p = [1.5 2.5 3.5; 4.5 5.5 6.5]
qp = (q = q, p = p)
turn_q_p_data_into_correct_format(qp)

# output

(q = [1.0 2.0; 4.0 5.0;;; 2.0 3.0; 5.0 6.0], p = [1.5 2.5; 4.5 5.5;;; 2.5 3.5; 5.5 6.5])
```
"""
function turn_q_p_data_into_correct_format(qp::QPT2{T, 2}) where {T}
	number_of_time_steps = size(qp.q, 2) - 1 # not counting t₀
	number_of_initial_conditions = size(qp.q, 1)
	q_array = zeros(T, 1, 2, number_of_time_steps * number_of_initial_conditions)
	p_array = zeros(T, 1, 2, number_of_time_steps * number_of_initial_conditions)
	for initial_condition_index ∈ 0:(number_of_initial_conditions - 1)
		for time_index ∈ 1:number_of_time_steps
			q_array[:, 1, initial_condition_index * number_of_time_steps + time_index] .= qp.q[initial_condition_index + 1, time_index]
			q_array[:, 2, initial_condition_index * number_of_time_steps + time_index] .= qp.q[initial_condition_index + 1, time_index + 1]
			p_array[:, 1, initial_condition_index * number_of_time_steps + time_index] .= qp.p[initial_condition_index + 1, time_index]
			p_array[:, 2, initial_condition_index * number_of_time_steps + time_index] .= qp.p[initial_condition_index + 1, time_index + 1]
		end
	end
	(q = q_array, p = p_array)
end

"""
This takes time as a single additional parameter (third axis).
"""
function load_time_dependent_harmonic_oscillator_with_parametric_data_loader(qp::QPT{T}, t::AbstractVector{T}, IC::AbstractVector) where {T}
	qp_reformatted = turn_q_p_data_into_correct_format(qp)
	t_reformatted = turn_parameters_into_correct_format(t, IC)
	ParametricDataLoader(qp_reformatted, t_reformatted)
end

"""
    forced_harmonic_oscillator_solution(t, IC; omega, Omega, F)

The analytic solution of the sinusoidally forced harmonic oscillator, as `(q, p)` arrays of size
`(number of initial conditions, length(t))`.
"""
function forced_harmonic_oscillator_solution(t::AbstractVector, IC::AbstractVector{<:NamedTuple};
		omega::Real, Omega::Real, F::Real)
	ni = length(IC)
	q = zeros(Float64, ni, length(t))
	p = zeros(Float64, ni, length(t))
	amplitude = F / (omega^2 - Omega^2)
	for i in eachindex(t)
		for j in 1:ni
			q[j, i] =  (IC[j].p - Omega * amplitude) / omega * sin(omega * t[i]) +
					   IC[j].q * cos(omega * t[i]) + amplitude * sin(Omega * t[i])
			p[j, i] = -omega^2 * IC[j].q * sin(omega * t[i]) +
					   (IC[j].p - Omega * amplitude) * cos(omega * t[i]) +
					   Omega * amplitude * cos(Omega * t[i])
		end
	end
	(q = q, p = p)
end
