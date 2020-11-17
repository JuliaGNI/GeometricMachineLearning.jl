using ForwardDiff

#define Hamiltonian
H(y) = .5 * sum(y[1,:].^2 + y[2,:].^2)

#compute differential equations  
dH(y) = [0 1;-1 0] * ForwardDiff.gradient(H,y)

#get data set:
num = 10
#range in which the data should be in
rang = range(-1.2,stop=1.2,length=num)
xy = [[x,y] for x in rang, y in rang]
x = reshape([point[1] for point in xy],num^2)
y = reshape([point[2] for point in xy],num^2)
dat = hcat(x,y)'
#compute the value of the vector field 
target = dH(dat)

#layer dimension
ld = 5

#activation functions
tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))
#relu(x) = max(0,x)

#build NN model
function model(x)
	y = x[1]; t = x[2];
	
	function loss(Wb)
		#Wb contains the weights for all the layers
		W1 = reshape(Wb[1:2*ld],ld,2); 
		b1 = Wb[2*ld+1:3*ld]; 
		W2 = reshape(Wb[3*ld+1:(3+ld)*ld],ld,ld); 
		b2 = Wb[(3+ld)*ld+1:(4+ld)*ld];
		W3 = reshape(Wb[(4+ld)*ld+1:(5+ld)*ld],1,ld);
		b3 = Wb[(5+ld)*ld+1];
		function est(τ) 
			#first layer
			layer1 = tanh.(W1 * τ .+ b1)
			#second layer
			layer2 = tanh.(W2 * layer1.+ b2)
			#third layer (linear activation)
			return (W3 * layer2 .+ b3)[1]
		end
		#compute vector for every element in the batch 
		vals = [[0 1; -1 0] * ForwardDiff.gradient(est,y[1:2,i]) for i in 1:size(y)[2]]
		#compute loss  
		return sum([sum((t[1:2,i] - vals[i]).^2) for i in 1:size(t)[2]])
	end	
end

#learning rate
η = .001

#initialise weights
Wb = randn((5+ld)*ld+1)

#make 100 learning runs
runs = 10000
arr_loss = zeros(runs)
total_loss = model((dat,target))
for j in 1:runs
	#select 10 points at random (one batch)
	local index = rand(1:num^2,10)
	local dat_loc = dat[1:2,index]
	local target_loc = target[1:2,index]
	local loss = model((dat_loc,target_loc))
	global Wb .-= η .* ForwardDiff.gradient(loss,Wb)	
	arr_loss[j] = total_loss(Wb)
end

#learned Hamiltonian
function H_est(τ)
	W1 = reshape(Wb[1:2*ld],ld,2);
	b1 = Wb[2*ld+1:3*ld];
	W2 = reshape(Wb[3*ld+1:(3+ld)*ld],ld,ld);
	b2 = Wb[(3+ld)*ld+1:(4+ld)*ld];
	W3 = reshape(Wb[(4+ld)*ld+1:(5+ld)*ld],1,ld);
	b3 = Wb[(5+ld)*ld+1];
	#first layer
	layer1 = tanh.(W1 * τ .+ b1)
	#second layer
	layer2 = tanh.(W2 * layer1.+ b2)
	#third layer (linear activation)
	return (W3 * layer2 .+ b3)[1]
end


##########all that follows are just diagnostics (especially plots)

#compute field
field_sc(τ) = [0 1; -1 0] * ForwardDiff.gradient(H_est,τ)
field(τ) = [[0 1; -1 0] * ForwardDiff.gradient(H_est,τ[1:2,i]) for i in 1:size(τ)[2]]

using Plots

#plot vector field
field_val = field(dat)
field_x = [field_val[i][1] for i in 1:num^2]
field_y = [field_val[i][2] for i in 1:num^2]
plt = quiver(dat[1,:],dat[2,:],quiver=(field_x,field_y))


#make on rk4 step
function rk4(y0, dt)
	dt2 = dt / 2.0
	k1 = field_sc(y0)
	k2 = field_sc(y0 + dt2*k1)  
	k3 = field_sc(y0 + dt2*k2)
	k4 = field_sc(y0 + dt	*k3)
  dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
  return dy
end 


#initial value(s) -> plot a few trajectories to check e-conservation
y0 = [[1,0],[0,1.2],[.5,0],[0,.25]]
steps = 2000
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


		
