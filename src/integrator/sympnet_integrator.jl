

struct SympNetMethod{TN <: LuxNeuralNetwork{<:SympNet}, tType <: Real} <: NeuralNetMethod 
    nn::TN
    Δt ::tType
end

function method(nns::NeuralNetSolution{<: LuxNeuralNetwork{<:SympNet}})
    SympNetMethod(nn(nns), tstep(nns))
end


const IntegratorSympNet{DT,TT} = Integrator{<:Union{HODEProblem{DT,TT}}, <:SympNetMethod}


function GeometricIntegrators.integrate(nns::NeuralNetSolution; kwargs...)
    integrate(problem(nns), method(nns); kwargs...)
end


function GeometricIntegrators.integrate_step!(int::IntegratorSympNet)

    # compute how may times to compose nn ()
    @assert  GeometricIntegrators.method(int).Δt % GeometricIntegrators.timestep(int) == 0 
    nb_comp =  GeometricIntegrators.method(int).Δt ÷ GeometricIntegrators.timestep(int)

    _q = solstep(int).q
    _p = solstep(int).p

    for _ in 1:nb_comp
        _qp = nn(GeometricIntegrators.method(int).nns)([_q...,_p...])
        _q, _p = (_qp[1::length(_q)], _qp[1+length(_q):end])
    end

    solstep(int).q = _q
    solstep(int).p = _p

end


