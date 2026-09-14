# Generates the `inverse_NxN.jl` files in `src/kernels/inverses/`. The script writes them into the
# working directory; run JuliaFormatter over the output with `style = "sciml"` and copy it into
# place. That reproduces the committed files byte for byte.
#
# `cse = true` is load-bearing. `build_function` defaults it to `false`, and each entry of the
# inverse then comes out as one deeply nested expression. JuliaFormatter indents every level of
# that nesting, and the cost is quadratic in the depth: formatted without CSE, the 5x5 file is
# 25 MB, of which 98 % is leading whitespace, and `fatou lint` does not terminate on it. With CSE
# the same inverse is a few hundred straight-line assignments and formats to tens of kilobytes.

using Symbolics

function matrix_inverse(n)
    @variables A[1:n, 1:n]

    B = inv(collect(A))

    if n ≤ 6
        B = simplify.(B)
    end

    build_function(B, A; cse = true)[2]
end

# `build_function` emits a plain `function (ˍ₋out, A)` over one n×n matrix. The kernels run over a
# tensor, one matrix per slice `k`, so every read of `A` gains the slice index and the linear
# output index becomes a Cartesian one.
function kernel_source(n)
    expr = matrix_inverse(n)
    str = replace(replace(replace(string(expr), r"#= [^*\s]* =#" => ""), r"\n[\s]*\n" => "\n"), "Num" => "")
    str = replace(str, r"var\"##cse#(\d+)\"" => s"t\1")
    str = replace(str, r"\(getindex\)\(A, (\d+), (\d+)\)" => s"(getindex)(A, \1, \2, k)")
    str = replace(str, r"ˍ₋out\[(\d+)\]" =>
        m -> begin
            l = parse(Int, match(r"\d+", m).match)
            "ˍ₋out[$(mod1(l, n)), $(cld(l, n)), k]"
        end)
    str = replace(str, r"^(\s*)ˍ₋out$"m => s"\1nothing")
    replace(str,
        "function (ˍ₋out, A)" => "@kernel function inv$(n)$(n)_kernel!(ˍ₋out, A)\n    k = @index(Global)")
end

function wrapper_source(n)
    """
    function tensor_inverse$(n)(A::AbstractArray{T, 3}) where {T}
        out = similar(A)

        tensor_inverse$(n)!(out, A)

        out
    end

    function tensor_inverse$(n)!(out::AbstractArray{T, 3}, A::AbstractArray{T, 3}) where {T}
        @assert size(A, 1) == size(A, 2) == $(n)
        @assert size(A) == size(out)

        backend = networkbackend(out)
        inv$(n)$(n)! = inv$(n)$(n)_kernel!(backend)

        inv$(n)$(n)!(out, A, ndrange = size(A, 3))

        nothing
    end

    function ChainRulesCore.rrule(::typeof(tensor_inverse$(n)), A::AT) where {
            T, AT <: AbstractArray{T, 3}}
        out = tensor_inverse$(n)(A)

        function tensor_inverse_pullback(out_diff)
            out_diff = unthunk(out_diff)

            NoTangent(),
            - tensor_transpose_tensor_mul(out, tensor_tensor_mul(out_diff, tensor_transpose(out)))
        end
        out, tensor_inverse_pullback
    end
    """
end

# The four sizes the package ships. `VolumePreservingAttention` dispatches on the sequence length
# and falls back to `cpu_tensor_cayley` for every other one.
for n in 2:5
    println("$(n)x$(n)...")
    write("inverse_$(n)x$(n).jl", kernel_source(n) * "\n\n" * wrapper_source(n))
end
