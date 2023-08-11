using Plots, JLD2, GeometricIntegrators, AbstractNeuralNetworks, GeometricMachineLearning

# this file stores parameters relevant for the NN
file_nn = jldopen("nn_model", "r")
model = file_nn["model"]
ps = file_nn["ps"]
seq_length = file_nn["seq_length"]
prediction_window = file_nn["prediction_window"]
transformer_dim = file_nn["transformer_dim"]
num_heads = file_nn["num_heads"]

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

model = Chain(  Dense(dim, transformer_dim, tanh),
              MultiHeadAttention(transformer_dim, num_heads),
              ResNet(transformer_dim, tanh),
              MultiHeadAttention(transformer_dim, num_heads),
              ResNet(transformer_dim, tanh),
              Dense(transformer_dim, dim, identity)
              )

T = Float64
m1 = T(2.)
m2 = T(1.)
k1 = T(1.5)
k2 = T(0.3)
k = T(3.5)
params = (m1=m1, m2=m2, k1=k1, k2=k2, k=k)
initial_conditions_val = (q=[T(1.),T(0.)], p=[T(2.),T(0.)])
t_integration = 50
pode = PODEProblem(q̇, ṗ, (T(0.0), T(t_integration)), time_step, initial_conditions_val; parameters = params)
sol = integrate(pode,ImplicitMidpoint())

data_matrix = zeros(T, 4, Int(t_integration/time_step) + 1)
for i in 1:seq_length data_matrix[1:2, i] = sol.q[i-1] end 
for i in 1:seq_length data_matrix[3:4, i] = sol.p[i-1] end

function compute_step(input::AbstractMatrix{T})
    model(input, ps)[:,(seq_length-prediction_window+1):end] 
end

total_steps = Int(floor((Int(t_integration/time_step) - seq_length + 1)/prediction_window))
for i in 1:total_steps
    data_matrix[:,(i-1)*prediction_window + seq_length + 1: i*prediction_window + seq_length] = compute_step(data_matrix[:,(i-1)*prediction_window+1:(i-1)*prediction_window+seq_length])
end

q1 = zeros(size(data_matrix,2))
t = zeros(size(data_matrix,2))
for i in 1:length(q1) 
    q1[i] = sol.q[i-1][1] 
    t[i] = sol.t[i-1]
end
plot1 = plot(t, q1, label="Numeric Integration", size=(1000,600))
plot!(plot1, t, data_matrix[1,:], label="Neural Network")
vline!(plot1, [seq_length*time_step], color="red",label="Start of Prediction")

png(plot1, "seq_length"*string(seq_length)*"_prediction_window"*string(prediction_window))