using Zygote
import Random 

Random.seed!(123)

function test_skew_symmetric_matrix_convergence(n::Int = 5, T::Type = Float32)
    A = rand(T, n, n)
    A = .5 * (A - A')
    ps = (weight = rand(SkewSymMatrix{T}, n), )
    o = Optimizer(AdamOptimizer(), ps)
    _loss(ps::NamedTuple, A::AbstractMatrix) = norm(ps.weight - A)
    for _ in 1:1200
        o.step += 1
        dp = Zygote.gradient(ps -> _loss(ps, A), ps)[1]
        update!(o, o.cache.weight, dp.weight)
        ps.weight .= ps.weight + dp.weight
    end
    # check if type stays SkewSymMatrix
    @test typeof(ps.weight) <: SkewSymMatrix
    @test ps.weight ≈ A
end

test_skew_symmetric_matrix_convergence()