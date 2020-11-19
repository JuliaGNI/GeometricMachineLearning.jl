using ModelingToolkit


#layer dimension/width
ld = 5

#size of Wb
Wb_siz = (5+ld)*ld+1


#variables are data, NNcoeff & targets
@variables y[1:2] Wb[1:Wb_siz] t[1:2]
@derivatives dy'~y dW'~Wb

#build neural network
W1 = reshape(Wb[1:2*ld],ld,2)
b1 = Wb[2*ld+1:3*ld];
W2 = reshape(Wb[3*ld+1:(3+ld)*ld],ld,ld);
b2 = Wb[(3+ld)*ld+1:(4+ld)*ld];
W3 = reshape(Wb[(4+ld)*ld+1:(5+ld)*ld],1,ld);
b3 = Wb[(5+ld)*ld+1];
#first layer
layer1 = atan.(W1 * y .+ b1)
#second layer
layer2 = atan.(W2 * layer1 .+ b2)
#output layer/estimate for Hamiltonian
est = (W3 * layer2 .+ b3)[1]


#compute vector field
vector_field = [0 1; -1 0] * ModelingToolkit.gradient(est,y)

loss = sum((vector_field-t).^2)

#gradient_step
step = ModelingToolkit.gradient(loss,Wb)

#build function 
step_fun = build_function(step,y,t,Wb)[1] 
write("step_fun.jl",string(step_fun))
write("loss.jl",string(build_function(loss,y,t,Wb)))



		
