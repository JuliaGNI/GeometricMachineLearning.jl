using BenchmarkTools
using Enzyme
using LinearAlgebra
using Zygote


loss(A,x) = norm(A*x)

# function loss(A,x)
#     y = zero(x)
#     mul!(y,A,x)
#     norm(y)
# end

function test(n)
    A = rand(n,n)
    x = rand(n)

    l = a -> loss(a,x)

    dA = zero(A)

    println("\nn = $n")

    println("\nEnzyme (autodiff):")
    @btime Enzyme.autodiff(Reverse, $l, Active, Duplicated($A, $dA))

    println("\nEnzyme (gradient):")
    @btime Enzyme.gradient(Reverse, $l, $A)

    println("\nZygote:")
    @btime Zygote.gradient($l, $A)[1]

    println("")
end

test(100)
test(1000)
test(10000)
