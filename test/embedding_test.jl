using Test
using Plots
using LaTeXStrings

include("../src/embeddings/sin_cos.jl")

function test_and_plot(λ::AbstractFloat, T::Int, e=3)

    Ξ =  SC_embed(λ, e, T)

    for i in 1:T
        for j in 1:e
            if iseven(j)
                @test abs(Ξ[i,j] - sin(i*λ^(j÷2))) < 1e-10 
            else
                @test abs(Ξ[i,j] - cos(i*λ^(j÷2))) < 1e-10
            end
        end
    end

    fig = Ξ_plot₂(Ξ)
    png(fig,"emb_la"*string(λ)*"_e"*string(e)*"_T"*string(T))

end

λ = [0.5, 0.3, 0.9]
T = [10, 20, 30]

for (λi, Ti) in zip(λ, T)
    test_and_plot(λi, Ti)
end 