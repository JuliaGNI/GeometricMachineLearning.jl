using Test

macro testerror(f, args...)
    error = false
    try 
        eval(f)(eval.(args)...)
    catch e
        error = true
    end
    @test error == true
end

macro testnoerror(f, args...)
    error = true
    try 
        eval(f)(eval.(args)...)
        error = false
    catch e
    end
    @test error == false
end