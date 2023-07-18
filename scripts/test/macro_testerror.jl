using Test

macro testerror(args...)
    error = false
    try 
        eval.(args)
    catch e
        error = true
    end
    @test error == true
end

macro testnoerror(args...)
    
    error = true
    try 
        eval.(args)
        error = false
    catch e
        println(e)
    end
    @test error == false
end
