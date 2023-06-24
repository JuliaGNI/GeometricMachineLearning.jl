using GeometricIntegrators

abstract type NeuralNetMethod <: GeometricMethod end


struct NeuralNetIntegrator{PT, MT, CT, SST} <: DeterministicIntegrator 
    problem::PT
    method::MT
    caches::CT
    solstep::SST

    function NeuralNetIntegrator(
        problem::AbstractProblem, 
        integratormethod::GeometricMethod;
        method = initmethod(integratormethod, problem),
        caches = CacheDict(problem, method),
        solstp = SolutionStep(problem, method)
        )

        new{typeof(problem),
            typeof(method),
            typeof(caches),
            typeof(solstp)
           }(problem, method, caches, solstp)
    end

end

Integrator(problem::AbstractProblem, method::NeuralNetMethod; kwargs...) = NeuralNetIntegratorIntegrator(problem, method; kwargs...)
Integrator(problem::AbstractProblem, nn::AbstractArchitecture; kwargs...) = NeuralNetIntegratorIntegrator(problem, method(nn); kwargs...)

method(nn::AbstractArchitecture) = @error "No integrator method assiociated to "*type_without_brace(nn)*"!"

problem(int::NeuralNetIntegrator) = int.problem
method(int::NeuralNetIntegrator) = int.method
caches(int::NeuralNetIntegrator) = int.caches
solstep(int::NeuralNetIntegrator) = int.solstep


GeometricBase.equations(int::NeuralNetIntegrator) = functions(problem(int))
GeometricBase.timestep(int::NeuralNetIntegrator) = timestep(problem(int))

#=
cache(int::NeuralNetIntegrator, DT) = caches(int)[DT]
cache(int::NeuralNetIntegrator) = cache(int, datatype(solstep(int)))
eachstage(int::NeuralNetIntegrator) = eachstage(method(int))
hasnullvector(int::NeuralNetIntegrator) = hasnullvector(method(int))
implicit_update(int::NeuralNetIntegrator) = implicit_update(method(int))
nconstraints(int::NeuralNetIntegrator) = nconstraints(problem(int))
Base.ndims(int::NeuralNetIntegrator) = ndims(problem(int))
nstages(int::NeuralNetIntegrator) = nstages(tableau(method(int)))
nlsolution(int::NeuralNetIntegrator) = nlsolution(cache(int))
nullvector(int::NeuralNetIntegrator) = nullvector(method(int))
tableau(int::NeuralNetIntegrator) = tableau(method(int))
=#


initial_guess!(::SolutionStep, ::AbstractProblem, ::GeometricMethod, ::CacheDict) = nothing
initial_guess!(int::NeuralNetIntegrator) = initial_guess!(solstep(int), problem(int), method(int), caches(int))
