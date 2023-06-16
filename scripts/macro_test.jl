macro testerror(f, args...)
    printstyled(center_align_text(string(eval(f)),17) * "|"; bold = true)
    for e in args
        if typeof(eval(e)) == Symbol
            text = string(eval(e))
        else
            text = print_type_without_brace(eval(e))
        end
        printstyled(center_align_text(text,20) * "|"; bold = true)
    end
    printstyled("|"; bold = true)

    try 
        eval(f)(eval.(args)...)
        printstyled(center_align_text("Passed",10); bold = true, color = :green)
        print("|\n")
    catch e
        printstyled(center_align_text("Failed",10); bold = true, color = :red)
        print(": ")
        printstyled(e; bold = true, color = :red)
        print("\n")
    end
end

macro testseterrors(exprs)
    
    #expression = filter_lineno(exprs)
    #exprs = filter(x-> x.head==:macrocall, expression.args)
    
    printstyled("Type of test:"; bold = true, underline = true) 
    printstyled(" ||"; bold = true) 
    printstyled(center_align_text("Integrator",20) * "|"; bold = true)
    printstyled(center_align_text("Type of Data",20) * "|"; bold = true)
    printstyled(center_align_text("Problem",20) * "|"; bold = true)
    printstyled(center_align_text("Optimiser",20) * "||"; bold = true)
    printstyled(center_align_text("Result",10) * "|\n"; bold = true)

    eval(exprs)

    return nothing
end

function center_align_text(text,width)
    padding = max(0, width - length(text))
    left_padding = repeat(" ",padding ÷2)
    right_padding = repeat(" ", padding - length(left_padding))
    aligned_text = left_padding * text * right_padding
    return aligned_text
end

function print_type_without_brace(var)
    type_str = string(typeof(var))
    replace(type_str, r"\{.*\}"=>"")
end


function filter_lineno(ex::Expr)
    filter!(ex.args) do e
        isa(e, LineNumberNode) && return false
        return true
    end
    return ex
end