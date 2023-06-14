# using Profile
using GeometricMachineLearning

# this contains the functions for generating the training data
include("../data_problem.jl")

macro testerror(f, args...)
    quote
        try 
            $(esc(:( $f($(args...)))))
            printstyled("Test Passed"; bold = true, color = :green)
            println()
        catch e
            printstyled("Test Failed"; bold = true, color = :red)
            println()
        end
    end
end


#HNN(integrator::Hnn_training_integrator, data::Training_data, nameproblem::Symbol = :pendulum, opt =  MomentumOptimizer(1e-3,0.5))



printstyled("Test of optimizer methods"; bold = true, underline = true)
println()


#@testerror HNN integrator data nameproblem opt
printstyled("Test of integrators and data"; bold = true, underline = true)
println()


printstyled("Test of differents problem"; bold = true, underline = true)
println()

