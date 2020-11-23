using ModelingToolkit

#layer dimension/width
ld = 5

#number of inputs/dimension of system
n_in = 2

#size of Wb: first layer + second layer + third layer
Wb_siz = (n_in*ld + ld) + (ld*ld + ld) + (ld + 1)

#variables are data, NNcoeff & targets
@variables y[1:n_in] Wb[1:Wb_siz] t[1:n_in]
@derivatives dy'~y dW'~Wb

#indices for Wb
i1 = n_in*ld
i2 = i1 + ld
i3 = i2 + ld*ld
i4 = i3 + ld
i5 = i4 + ld

#build neural network
W1 = reshape(Wb[1:i1], ld, n_in)
b1 = Wb[i1+1:i2];
W2 = reshape(Wb[i2+1:i3], ld, ld);
b2 = Wb[i3+1:i4];
W3 = reshape(Wb[i4+1:i5],1,ld);
b3 = Wb[i5+1];
#first layer
layer1 = Base.tanh.(W1 * y .+ b1)
#second layer
layer2 = Base.tanh.(W2 * layer1 .+ b2)
#output layer/estimate for Hamiltonian
est = sum(W3 * layer2 .+ b3)


#compute vector field
vector_field = [0 1; -1 0] * ModelingToolkit.gradient(est,y)

loss = sum((vector_field-t).^2)

#gradient_step
step = ModelingToolkit.gradient(loss,Wb)

#build functions 
step_fun = build_function(step,y,t,Wb)[1]
field = build_function(vector_field,y,t,Wb)[1]


#####save functions -> mt_fun ... ModelingToolkit functions 
write("step_fun.jl",string(step_fun))
write("field.jl",string(field))

#scalar functions
write("loss.jl",string(build_function(loss,y,t,Wb)))
write("est.jl",string(build_function(est,y,t,Wb)))


		
