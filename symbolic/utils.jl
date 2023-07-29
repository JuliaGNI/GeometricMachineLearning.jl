function get_string(expr)
    replace(replace(replace(string(expr), r"#= [^*\s]* =#" => ""), r"\n[\s]*\n" => "\n"), "Num" => "")
end

develop(x) = (x,)
develop(t::Tuple) = Tuple(vcat([[develop(e)...] for e in t]...))
develop(t::NamedTuple) = Tuple(vcat([[develop(e)...] for e in t]...))


macro Name(x)
    string(x)
end

function SymbolicName(arg, i)
    nam = string(arg)
    i != 0 ? nam *= "_"*string(i) :  nothing
    Symbol(nam)
end


function transposymplecticMatrix(n::Int) 
    I = Diagonal(ones(n÷2))
    Z = zeros(n÷2,n÷2)
    [Z -I; I Z]
end

