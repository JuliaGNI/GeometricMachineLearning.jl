using GeometricIntegrators, KernelAbstractions

# here the second point mass is altered
params_collection = (  (m1=2, m2=2, k1=1.5, k2=0.1, k=0.2),
            (m1=2, m2=1, k1=1.5, k2=0.1, k=0.2),
            (m1=2, m2=.5, k1=1.5, k2=0.1, k=0.2),
            (m1=2, m2=.25, k1=1.5, k2=0.1, k=0.2)
)

initial_conditions_collection = ( (q=[1.,0.], p=[2.,0.]),
                    (q=[1.,0.], p=[1.,0.]),
                    (q=[1.,0.], p=[0.5,0.]))

t_integration = 1000

function q̇(v, t, q, p, params)
    v[1] = p[1]/params.m1
    v[2] = p[2]/params.m2
end

function ṗ(f, t, q, p, params)
    f[1] = -params.k1 * q[1] - params.k * (cos(10*q[1]) + 1) * (q[1] - q[2]) + params.k /2 * (q[1] - q[2])^2 * 10 * sin(10 * q[1])
    f[2] = -params.k2 * q[2] + params.k * (cos(10*q[1]) + 1) * (q[1] - q[2]) 
end

sols = []
for params in params_collection
    for initial_conditions in initial_conditions_collection
        pode = PODEProblem(q̇, ṗ, (0.0, t_integration), .1, initial_conditions; parameters = params)
        sol = integrate(pode,ImplicitMidpoint())
        push!(sols, sol)
    end
end

time_steps = length(sols[1].q)
data_tensor = zeros(4, length(sols), time_steps)
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

function generate_data()
    sols = []
    for params in params_collection
        for initial_conditions in initial_conditions_collection
            pode = PODEProblem(q̇, ṗ, (0.0, t_integration), .1, initial_conditions; parameters = params)
            sol = integrate(pode,ImplicitMidpoint())
            push!(sols, sol)
        end
    end

    time_steps = length(sols[1].q)
    data_tensor = zeros(4, length(sols), time_steps)
    assign_tensor(data_tensor, sols)
    data_tensor 
end