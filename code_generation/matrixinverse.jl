using Symbolics

function matrix_inverse(n)
    @variables A[1:n,1:n]

    B = inv(collect(A))
    
    if n ≤ 6
        B = simplify.(B)
    end
    
    build_function(B, A)[2]
end

for n in 2:8
    println("$(n)x$(n)...")
    expr = matrix_inverse(n)
    str = replace(replace(replace(string(expr), r"#= [^*\s]* =#" => ""), r"\n[\s]*\n" => "\n"), "Num" => "")
    str = replace(str, "function (ˍ₋out, A)" => "function matinv$(n)x$(n)(ˍ₋out, A)")
    write("matrixinverse_$(n)x$(n).jl", str)
end
