using ForwardDiff

#define Hamiltonian
H(y) = .5 * sum(y[1,:].^2 + y[2,:].^2)

#compute differential equations  
dH(y) = [0 1;-1 0] * ForwardDiff.gradient(H,y)


#get data set:
num = 10
#the range in which the data will be collected
rang = range(-1.2,stop=1.2,length=num)
xy = [[x,y] for x in rang, y in rang]
x = reshape([point[1] for point in xy],num^2)
y = reshape([point[2] for point in xy],num^2)
dat = hcat(x,y)'
#compute the vector field at the target points
target = dH(dat)

#layer dimension/width
ld = 10

#activation functions
tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))
#relu(x) = max(0,x)

#rudimentary NN model
function model(x)
	y = x[1]; t = x[2];
	
	function loss(Wb)
		#Wb contains the weights for all the layers
		W1 = reshape(Wb[1:2*ld],ld,2); 
		b1 = Wb[2*ld+1:3*ld]; 
		W2 = reshape(Wb[3*ld+1:(3+ld)*ld],ld,ld); 
		b2 = Wb[(3+ld)*ld+1:(4+ld)*ld];
		W3 = reshape(Wb[(4+ld)*ld+1:(6+ld)*ld],2,ld);
		b3 = Wb[(6+ld)*ld+1:(6+ld)*ld+2];
		#first layer
		layer1 = tanh.(W1 * y .+ b1)
		#second layer
		layer2 = tanh.(W2 * layer1.+ b2)
		#third layer (linear activation)
		vals =  W3 * layer2 .+ b3
		return sum((t .- vals).^2)
	end
		
end

#learning rate
η = .001

#initialise weights
Wb = 10*randn((6+ld)*ld+2)

#make 10000 learning runs
runs = 10000
arr_loss = zeros(runs)
for j in 1:runs
	#select 100 points at random (one batch)
	local index = rand(1:num^2,10)
	local dat_loc = dat[1:2,index]
	local target_loc = target[1:2,index]
	local loss = model((dat_loc,target_loc))
	global Wb .-= η .* ForwardDiff.gradient(loss,Wb)
	local loss = model((dat,target))	
	arr_loss[j] = loss(Wb)
end

#calculate the learned field
function field(y)
	W1 = reshape(Wb[1:2*ld],ld,2);
	b1 = Wb[2*ld+1:3*ld];
	W2 = reshape(Wb[3*ld+1:(3+ld)*ld],ld,ld);
	b2 = Wb[(3+ld)*ld+1:(4+ld)*ld];
	W3 = reshape(Wb[(4+ld)*ld+1:(6+ld)*ld],2,ld);
	b3 = Wb[(6+ld)*ld+1:(6+ld)*ld+2];
	#first layer
	layer1 = tanh.(W1 * y .+ b1)
	#second layer
	layer2 = tanh.(W2 * layer1.+ b2)
	#third layer (linear activation)
	return W3 * layer2 .+ b3
end

#####diagnostics ... mainly plots

using Plots

#plot vector field
quiver(dat[1,:],dat[2,:],quiver=(field(dat)[1,:],field(dat)[2,:]))

#function that makes one rk4 step
function rk4(y0, dt)
        dt2 = dt / 2.0
        k1 = field(y0)
        k2 = field(y0 + dt2*k1)
        k3 = field(y0 + dt2*k2)
        k4 = field(y0 + dt   *k3)
  dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
  return dy
end

#initial value(s)
y0 = [1,0]
steps = 1000
dt = 2*π/steps
p = zeros(steps); q = zeros(steps)
p[1] = y0[1]; q[1] = y0[2]

for i=2:steps
        p[i],q[i] = [p[i-1],q[i-1]] + rk4([p[i-1],q[i-1]],dt)
end

## the "closedness" of the curve indicates a conservative field
plot!(p,q)
	
