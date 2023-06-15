# using Profile
using GeometricMachineLearning

macro test_error(f,g)
    println(g,f)
    
end

function test(x)
    @assert !(x<0) "x must be possitive"
    x>1 || @warn "x plu grand que 1"
    nothing
end
