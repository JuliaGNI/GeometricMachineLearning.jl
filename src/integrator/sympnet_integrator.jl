

struct SympNetMethod{TN <: NeuralNetwork{<:SympNet}, tType <: Real} <: NeuralNetMethod 
    nn::TN
    Δt ::tType
end

function method(nns::NeuralNetSolution{<: NeuralNetwork{<:SympNet}})
    SympNetMethod(nn(nns), tstep(nns))
end


function GeometricIntegrators.Integrators.integrate(nns::NeuralNetSolution; kwargs...)
    integrate(problem(nns), method(nns); kwargs...)
end


function GeometricIntegrators.Integrators.integrate_step!(int::GeometricIntegrator{<:SympNetMethod, <:AbstractProblemPODE})

    # compute how may times to compose nn ()
    @assert  GeometricIntegrators.Integrators.method(int).Δt % GeometricIntegrators.timestep(int) == 0 
    nb_comp =  method(int).Δt ÷ GeometricIntegrators.timestep(int)

    _q = GeometricIntegrators.Integrators.solstep(int).q
    _p = GeometricIntegrators.Integrators.solstep(int).p

    for _ in 1:nb_comp
        _qp = (GeometricIntegrators.Integrators.method(int).nn)([_q...,_p...])
        _q, _p = (_qp[1:length(_q)], _qp[(1+length(_q)):end])
    end

    GeometricIntegrators.Integrators.solstep(int).q = StateVariable(_q)
    GeometricIntegrators.Integrators.solstep(int).p = StateVariable(_p)

end
