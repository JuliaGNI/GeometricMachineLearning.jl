#get data set
include("plots_data/data.jl")
dat, target = get_data_set()

#load step and loss functions
using ModelingToolkit
step = include("mt_fun/step_fun.jl")
loss = include("mt_fun/loss.jl")

#number of inputs/dimension of system
n_in = 2

#perform one training step; η ... learning rate
function training_step(Wb, dat_set, target_set, batch_size=10, η=.001)
	index = rand(axes(dat_set,2), batch_size)
	Wb .-= η .* sum([step(dat_set[1:n_in,i], target[1:n_in,i],Wb) for i in index])
end


#layer dimension/width
ld = 5
#size of Wb: first layer + second layer + third layer
Wb_siz = (n_in*ld + ld) + (ld*ld + ld) + (ld + 1)

#initialise Wb
Wb = randn(Wb_siz)


#make 2000 learning runs
runs = 2000
arr_loss = zeros(runs)
for j in 1:runs
	training_step(Wb, dat, target) 
	arr_loss[j] = sum([loss(dat[1:n_in,i],target[1:n_in,i],Wb) for i in axes(dat,2)])
end

#get Hamiltonian (second argument is dummy target value for variable t)
est = include("mt_fun/est.jl")
H_est(τ) = est(τ, zeros(2), Wb)


#get vector field (second argument is dummy target value)
field_fun = include("mt_fun/field.jl")
field_sc(τ) = field_fun(τ, zeros(2), Wb)
field(τ) = [field_sc(τ[1:n_in,i]) for i in axes(τ,2)]

##########all that follows are just diagnostics (especially plots)
include("plots_data/plots.jl")


