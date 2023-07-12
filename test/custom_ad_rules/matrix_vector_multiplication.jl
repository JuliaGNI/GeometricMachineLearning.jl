#example taken from https://juliadiff.org/ChainRulesCore.jl/stable/rule_author/example.html
using ChainRulesCore

struct Foo{T}
    A::Matrix{T}
    c::Float64
end

function foo_mul(foo::Foo, b::AbstractArray)
    return foo.A * b
end

#the @thunk macro means that the computation is only performed in case it is needed
function ChainRulesCore.rrule(::typeof(foo_mul), foo::Foo{T}, b::AbstractArray) where T
    y = foo_mul(foo, b)
    function foo_mul_pullback(ȳ)
        f̄ = NoTangent()
        f̄oo = @thunk Tangent{Foo{T}}(; A=ȳ * b', c=ZeroTangent())
        b̄ = @thunk foo.A' * ȳ
        return f̄, f̄oo, b̄
    end
    return y, foo_mul_pullback
end

#rrule is compared to finite differences (FiniteDifferences.jl) 
using ChainRulesTestUtils
test_rrule(foo_mul, Foo(rand(3, 3), 3.0), rand(3, 3))
