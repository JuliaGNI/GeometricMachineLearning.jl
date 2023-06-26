"""
Implements a positional embedding for transformers as in the original paper (Attention is all you need: arXiv:1706.03762) with sines and cosines:
P_{i,2j} = 	sin(i/10000^{2j/K}) = sin(i⋅λʲ),
P_{i,2j+1} =cos(i/10000^{2j/K}) = cos(i⋅λʲ) with λ(K) ≡ λ := 10000^{-2/K} = 10^{-8/K}.
The bigger K, the smaller the decay of the frequency; e.g. λ(20) ≈ 0.5.

The mapping can also be seen as: 
| 1	|	| 1  λ   λ   λ²   λ²  ⋯  λ^{(e-1)/2} λ^{(e-1)/2}   |
| ⋮	|	| ⋯  ⋯  ⋯   ⋯   ⋯   ⋯   ⋯  ⋯   ⋯   ⋯   			   |
| i	| ↦ | i λ⋅i λ⋅i λ²⋅i λ²⋅i ⋯ λ^{(e-1)/2}⋅i λ^{(e-1)/2}⋅i| = Mat₁(λ,e,T)
| ⋮	|	| ⋯  ⋯  ⋯   ⋯   ⋯   ⋯   ⋯  ⋯   ⋯   ⋯   			   |
| T |	| T λ⋅T λ⋅T λ²⋅T λ²⋅T ⋯ λ^{(e-1)/2}⋅T λ^{(e-1)/2}⋅T|
composed with another mapping λ^{j÷2}⋅i ↦ sin(λ^{j÷2}⋅i) (if j even); or cos(λ^{j÷2}⋅i) (if j odd).

The first index is the row index, the second one the column index.
K ... context window
e ... embedding dimension
T ... number of times steps
"""

λ_con(K::Integer, T=Float32) = T(10.) ^(T(-8.) * T(inv(K)))

function sc_embed(λ::T, dim::Integer, t_final::Integer) where T
	mat = zeros(T, dim, t_final)
	for i=1:dim
		for pos=1:t_final
			if iseven(i)
				mat[i,pos] = sin(pos*λ^(i÷2))
			else
				mat[i,pos] = cos(pos*λ^(i÷2))
			end
		end
	end
	mat
end
sc_embed(dim::Integer, t_final::Integer, T=Float32) = sc_embed(λ_con(dim, T), dim, t_final)

function sc_embed(A::AbstractMatrix{T}) where T
	A + sc_embed(size(A)..., T)
end

#don't need quiver for any of this stuff!!!!
#=
function Ξ_plot!(fig, Ξ; args...)
	T, e = size(Ξ)
    #convert into the right format for quiver, Ξᵥ = [col₁, col₂, …, colₑ]
    Ξᵥ = []
	for i in 1:e push!(Ξᵥ,Ξ[:,i]) end
	#convert to tuple for quiver
	Ξᵥ = Tuple(Ξᵥ)
    zero_vec = [zeros(T) for i in 1:e]
    quiver!(fig, zero_vec..., quiver = Ξᵥ; args...)
end

function Ξ_plot(Ξ; args...)
	fig = quiver()
	Ξ_plot!(fig, Ξ; args...)
	return fig
end

#chance the top of the quiver plot (i.e. Ξ[i,:])
function Ξ_plot₂(Ξ)
	#convert to [col₁, …, colₑ] 
	Ξᵥ = []
	e = size(Ξ)[2]
	for i in 1:e push!(Ξᵥ,Ξ[:,i]) end
	fig = plot!(Ξᵥ...; color=1, label=nothing)
	fig = Ξ_plot!(fig, Ξ[1,:]'; color=1)
	plot!(fig, [0], [0], [0]; color=1, label=L"\xi"*"1")
	for i in 2:(size(Ξ)[1]) 
		Ξ_plot!(fig,Ξ[i,:]'; color=i)
		plot!(fig, [0], [0], [0]; color=i, label=L"\xi"*string(i)) 
	end
	return fig
end
=#
