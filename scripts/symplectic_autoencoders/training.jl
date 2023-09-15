using GeometricMachineLearning 
using LinearAlgebra: svd, norm
using ProgressMeter
using Zygote
include("generate_data.jl")

T = Float32
spacing=T(.01)
time_step=T(0.01)
μ_collection=T(5/12):T(.1):T(5/6)
n = 5
n_epochs = 2000
backend = CPU()

data = generate_data(T;spacing=spacing,time_step=time_step,μ_collection=μ_collection)
data = reshape(data, size(data,1), size(data,2)*size(data,3))
N = size(data,1)÷2
dl = DataLoader(data)
Φ = svd(hcat(data[1:N,:], data[(N+1):2*N,:])).U[:,1:n]
PSD = hcat(vcat(Φ, zero(Φ)), vcat(zero(Φ), Φ))
PSD_error = norm(data - PSD*PSD'*data)/norm(data)

activation = tanh
model = Chain(  GradientQ(2*N, 2*N, activation), 
                GradientP(2*N, 2*N, activation),
                PSDLayer(2*N, 2*n),
                GradientQ(2*n, 2*n, activation),
                GradientP(2*n, 2*n, activation),
                GradientQ(2*n, 2*n, activation),
                GradientP(2*n, 2*n, activation),
                PSDLayer(2*n, 2*N),
                GradientQ(2*N, 2*N, activation),
                GradientP(2*N, 2*N, activation)
)

ps = initialparameters(backend, Float32, model)
loss(model, ps, dl)

optimizer_instance = Optimizer(AdamOptimizer(), ps)
n_training_iterations = Int(ceil(n_epochs*dl.n_params/dl.batch_size))
progress_object = Progress(n_training_iterations; enabled=true)

for _ in 1:n_training_iterations
    redraw_batch!(dl)
    loss_val, pb = Zygote.pullback(ps -> loss(model, ps, dl), ps)
    dp = pb(one(loss_val))[1]

    optimization_step!(optimizer_instance, model, ps, dp)
    ProgressMeter.next!(progress_object; showvalues=[(:TrainingLoss, loss_val)])
end