using GeometricMachineLearning, ProgressMeter, Zygote
using KernelAbstractions
using CUDA
using Random
using JLD2

backend = CPU()
T = Float32

file = jldopen("data", "r")
data_raw = file["tensor"]
sys_dim, n_params, n_time_steps = size(data_raw)

data = KernelAbstractions.allocate(backend, T, size(data_raw))
copyto!(data, data_raw)

transformer_dim = 20
num_heads = 4
seq_length = 20
n_epochs = 2000
batch_size = 512
prediction_window = 5
include("auxiliary_functions.jl")

model = Chain( PSDLayer(sys_dim, transformer_dim),
              Attention(transformer_dim),
              Gradient(transformer_dim, 2*transformer_dim, tanh),
              Gradient(transformer_dim, 2*transformer_dim, tanh, change_q=false),
              Attention(transformer_dim),
              Gradient(transformer_dim, 2*transformer_dim, tanh, change_q=false),
              Gradient(transformer_dim, 2*transformer_dim, tanh),
              PSDLayer(transformer_dim, sys_dim)
              )

loss(ps) = loss(model, ps)
ps = initialparameters(backend, T, model)

o = Optimizer(AdamOptimizer(), ps)

n_training_steps_per_epoch = Int(ceil(n_time_steps/batch_size))
n_training_steps = n_epochs*n_training_steps_per_epoch

progress_object = Progress(n_training_steps; enabled=true)

for t in 1:n_training_steps
    draw_batch!(batch, output, seq_length, prediction_window)
    loss_val, pullback = Zygote.pullback(loss, ps)
    dx = pullback(1)[1]
    optimization_step!(o, model, ps, dx)
    ProgressMeter.next!(progress_object; showvalues = [(:TrainingLoss,loss_val)])
end

map_to_cpu(ps::Tuple) = Tuple([map_to_cpu(layer) for layer in ps])
map_to_cpu(layer::NamedTuple) = apply_toNT(map_to_cpu, layer)
function map_to_cpu(A::AbstractArray{T}) where T
    Array{T}(A)
end

jldsave("nn_model_symplectic", model=model, ps=ps, seq_length=seq_length, prediction_window=prediction_window, transformer_dim=transformer_dim, num_heads=num_heads)
