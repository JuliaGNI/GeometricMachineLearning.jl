using Plots, JLD2, GeometricIntegrators

# this file stores parameters relevant for the NN
file_nn = jldopen("nn_model", "r")
model = file_nn["model"]
ps = file_nn["ps"]
seq_length = file_nn["seq_length"]
prediction_window = file_nn["prediction_window"]

# this file stores parameters relevant for dataset/vector field (theoretically only needed for validation!)
file_data = jldopen("data", "r")

# somehow find a routine so that you don't have to define the vector field twice
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
time_step = file_data["time_step"]

T = Float64
m1 = T(2.)
m2 = T(1.)
k1 = T(1.5)
k2 = T(0.3)
k = T(4.5)
params = (m1=m1, m2=m2, k1=k1, k2=k2, k=k)
initial_conditions_val = (q=[T(1.),T(0.)], p=[T(2.),T(0.)])
t_integration = 100
pode = PODEProblem(q̇, ṗ, (T(0.0), T(t_integration)), time_step, initial_conditions_val; parameters = params)
sol = integrate(pode,ImplicitMidpoint())

data_matrix = zeros(T, 4, Int(t_integration/time_step) + 1)
#data_