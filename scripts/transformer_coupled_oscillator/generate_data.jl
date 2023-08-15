using GeometricIntegrators, KernelAbstractions, JLD2

T = Float64

initial_conditions_collection = ( (q=[T(1.),T(0.)], p=[T(2.),T(0.)]), )

m1 = T(2.)
m2 = T(1.)
k1 = T(1.5)
k2 = T(0.3)
k = T.(0.0:0.01:4)

params_collection = Tuple([(m1=m1,m2=m2,k1=k1,k2=k2,k=k_val) for k_val in k])

const t_integration = 1000
const time_step = T(.4)

function q̇(v, t, q, p, params)
    v[1] = p[1]/params.m1
    v[2] = p[2]/params.m2
end

function sigmoid(x::T) where {T<:Real}
	T(1)/(T(1) + exp(-x))
end

function ṗ(f, t, q, p, params)
	f[1] = -params.k1 * q[1] - params.k * (q[1] - q[2]) * sigmoid(q[1]) - params.k /2 * (q[1] - q[2])^2 * sigmoid(q[1])^2 * exp(-q[1])
	f[2] = -params.k2 * q[2] + params.k * (q[1] - q[2]) * sigmoid(q[1])
end

@kernel function create_tensor_kernel_q!(data_tensor, sols)
    i,j,k = @index(Global, NTuple)
    data_tensor[i,j,k] = sols[j].q[k-1][i]
end
@kernel function create_tensor_kernel_p!(data_tensor, sols)
    i,j,k = @index(Global, NTuple)
    data_tensor[i+2,j,k] = sols[j].p[k-1][i]
end
function assign_tensor(data_tensor, sols)
    assign_q! = create_tensor_kernel_q!(CPU())
    assign_p! = create_tensor_kernel_p!(CPU())
    dims = (2, size(data_tensor,2), size(data_tensor,3))
    assign_q!(data_tensor, sols, ndrange=dims)
    assign_p!(data_tensor, sols, ndrange=dims)
end

function generate_data(params_collection=params_collection, initial_conditions=initial_conditions, t_integration=t_integration, Float32_not_working_yet=true)
    Float32_not_working_yet ? (@warn "Float32 not used in integration!!") : nothing
    sols = []
    for params in params_collection
        for initial_conditions in initial_conditions_collection
            pode = PODEProblem(q̇, ṗ, (0.0, t_integration), time_step, initial_conditions; parameters = params)
            sol = integrate(pode,ImplicitMidpoint())
            push!(sols, sol)
        end
    end

    time_steps = length(sols[1].q)
    data_tensor = zeros(4, length(sols), time_steps)
    assign_tensor(data_tensor, sols)
    Float32_not_working_yet ? Float32.(data_tensor) : data_tensor
end

data_tensor = generate_data()

jldsave("data", tensor=data_tensor, params=params_collection, initial_conditions=initial_conditions_collection, 
        time_interval=(0, t_integration), time_step=time_step)