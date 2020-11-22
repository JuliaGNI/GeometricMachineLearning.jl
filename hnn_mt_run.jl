using ModelingToolkit

@variables y[1:2]

#get data set
include("data.jl")
dat, target = get_data_set()


#learning rate
η = .001

#layer dimension/width
ld = 5
#size of Wb
Wb_siz = (5+ld)*ld+1
#initialise Wb
Wb = randn(Wb_siz)

#load step and loss functions
step = include("step_fun.jl")
loss = include("loss.jl")

#make 2000 learning runs
runs = 2000
batch_size = 10
arr_loss = zeros(runs)
for j in 1:runs 
        #select 10 points at random (one batch)
        local index = rand(axes(dat,2),batch_size)
        global Wb .-= η .* sum([step(dat[1:2,i],target[1:2,i],Wb) for i in index])
        arr_loss[j] = sum([loss(dat[i],target[i],Wb) for i in index])
end

#compute Hamiltonian
W1 = reshape(Wb[1:2*ld],ld,2)
b1 = Wb[2*ld+1:3*ld];
W2 = reshape(Wb[3*ld+1:(3+ld)*ld],ld,ld);
b2 = Wb[(3+ld)*ld+1:(4+ld)*ld];
W3 = reshape(Wb[(4+ld)*ld+1:(5+ld)*ld],1,ld);
b3 = Wb[(5+ld)*ld+1];
#first layer
layer1(τ) = atan.(W1 * τ .+ b1)
#second layer
layer2(τ) = atan.(W2 * layer1(τ) .+ b2)
#output layer/estimate for Hamiltonian
H_est(τ) = (W3 * layer2(τ) .+ b3)[1]


#compute vector field
vector_field_sym = [0 1; -1 0] * ModelingToolkit.gradient(H_est(y),y)
vector_field = build_function(vector_field_sym,y)[1] |> eval

##########all that follows are just diagnostics (especially plots)

using Plots

#plot vector field
dat_x = [dat[1,i] for i in axes(dat,2)]
dat_y = [dat[2,i] for i in axes(dat,2)]
field_val = [vector_field(dat[1:2,i]) for i in axes(dat,2)]
field_x = [field_val[i][1] for i in axes(dat,2)]
field_y = [field_val[i][2] for i in axes(dat,2)]
plt = quiver(dat_x,dat_y,quiver=(field_x,field_y))


#make on rk4 step
function rk4(y0, dt)
        dt2 = dt / 2.0
        k1 = vector_field(y0)
        k2 = vector_field(y0 + dt2*k1)
        k3 = vector_field(y0 + dt2*k2)
        k4 = vector_field(y0 + dt *k3)
  dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
  return dy
end


#initial value(s) -> plot a few trajectories to check e-conservation
y0 = [[1,0],[0,1.2],[.5,0],[0,.25],[0,1.4]]
#number of steps for rk4
steps = 2000
#one period is π long - so make 4 turns
dt = 4*π/steps


for k in y0
        global p = zeros(steps); global q = zeros(steps)
        p[1] = k[1]; q[1] = k[2]
        #make rk4 steps
        for i=2:steps
                p[i],q[i] = [p[i-1],q[i-1]] + rk4([p[i-1],q[i-1]],dt)
        end

        #closed curve indicates a conservative field
        display(plot!(p,q))
end

#check energy conservation
arr = [H_est([p[i],q[i]]) for i in 1:steps]


