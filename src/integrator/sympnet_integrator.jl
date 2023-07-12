

struct SympNetMethod{TN <: NeuralNetSolution{<: LuxNeuralNetwork{<:SympNet}}} <: NeuralNetMethod 
    nns::TN
end

function method(nn::LuxNeuralNetwork{<:SympNet})
    function nnₛₚₗᵢₜ(q,p)
        qp = nn([q...,p...])
        (qp[1::length(q)], qp[1+length(q):end])
    end
    SympNetMethod(nnₛₚₗᵢₜ)
end

const IntegratorSympNet{DT,TT} = Integrator{<:Union{HODEProblem{DT,TT}}, <:SympNetMethod}

function integrate_step!(int::IntegratorSympNet)

    # compute how may times to compose nn ()
    @assert tstep(problem(int)) % timestep(method(int)) == 0 
    nb_comp = tstep(problem(int)) ÷ timestep(method(int))

    _q = solstep(int).q
    _p = solstep(int).p

    for _ in 1:nb_comp
        _q, _p = nn(method(int).nns)(_q,_p)
    end

    solstep(int).q = _q
    solstep(int).p = _p
end


