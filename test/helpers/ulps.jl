# A recorded vector is compared at a stated count of units in the last place of the larger
# magnitude. A value that passes through GEMM and `tanh` is not bit-identical across CPUs and BLAS
# builds: one recorded vector differs by 1 to 2 ULPs between x86_64 and arm64, while a change of an
# extracted layer or parameter set moves it by a fraction, thousands of ULPs away.
#
# `eps(x)` is the spacing of floats at `x`'s magnitude, so `n * eps(max(abs(x), abs(y)))` is `n`
# ULPs there.
function at_ulps(a, b; n::Integer = 8)
    length(a) == length(b) || return false
    for (x, y) in zip(a, b)
        x == y && continue
        abs(x - y) <= n * eps(max(abs(x), abs(y))) || return false
    end
    return true
end
