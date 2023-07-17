"""
Implements a positional embedding for transformers as in the original paper (Attention is all you need: arXiv:1706.03762) with sines and cosines:
P_{i,2j} = 	sin(i/10000^{2j/K}) = sin(i⋅λʲ),
P_{i,2j+1} =cos(i/10000^{2j/K}) = cos(i⋅λʲ) with λ(K) ≡ λ := 10000^{-2/K} = 10^{-8/K}.
The bigger K, the smaller the decay of the frequency; e.g. λ(20) ≈ 0.5.

The first index is the row index, the second one the column index.
K ... context window
e ... embedding dimension
T ... number of times steps
"""
function sc_embed(A::AbstractMatrix{T}, λ::T) where T
	mat = copy(A)
	sc_embed!(mat, λ)
	mat
end

λ_fac(K::Integer, T=Float32) = T(10.) ^(T(-8.) * T(inv(K)))

function sc_embed!(A::AbstractMatrix{T}, λ::T) where T
	dim, t_final = size(A)
	for i=1:dim
		for pos=1:t_final
			if iseven(i)
				A[i,pos] += sin(pos*λ^(i÷2))
			else
				A[i,pos] += cos(pos*λ^(i÷2))
			end
		end
	end
end

function sc_embed!(A::AbstractMatrix{T}) where T 
	sc_embed!(A, λ_fac(size(A,2), T))
end

function sc_embed(A::AbstractMatrix)
	mat = copy(A)
	sc_embed!(mat)
	mat 
end


