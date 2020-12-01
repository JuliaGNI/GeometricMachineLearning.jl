using ModelingToolkit

#settings and common functionality
include("common.jl")

#this file contains the function that evaluates the network
include("../utils/networks.jl")

#generate code for neural network
function generate_mt_hnn(n_in, ld)
    #input and training variables
    @variables y[1:n_in]  t[1:n_in]

    #initialise weights
    @variables W1[1:ld, 1:n_in]  b1[1:ld]
    @variables W2[1:ld, 1:ld]    b2[1:ld]
    @variables W3[1:1,  1:ld]

	model = (
		(W = W1, b = b1),
		(W = W2, b = b2),
		(W = W3,       ),
    )

    #derivatives
    @derivatives dy'~y
    @derivatives dW1'~W1  db1'~b1
    @derivatives dW2'~W2  db2'~b2
    @derivatives dW3'~W3

    #output layer/estimate for Hamiltonian
    est = sum(network(y, model))

    #compute vector field
    field = [0 1; -1 0] * ModelingToolkit.gradient(est, y)

	#compute loss  
    loss = sum((field .- t).^2)

    #gradient_step
    step = (
        (W = reshape(ModelingToolkit.gradient(loss, vec(W1)), 1:ld, 1:n_in), b = ModelingToolkit.gradient(loss, b1)),
        (W = reshape(ModelingToolkit.gradient(loss, vec(W2)), 1:ld, 1:ld),   b = ModelingToolkit.gradient(loss, b2)),
        (W = reshape(ModelingToolkit.gradient(loss, vec(W3)), 1:1,  1:ld),   ),
    )

    #build functions 
    fun_est   = build_function(est,   y,    expand(model)...)
    fun_field = build_function(field, y,    expand(model)...)[1]
    fun_loss  = build_function(loss,  y, t, expand(model)...)
    fun_step  = build_function(step,  y, t, expand(model)...)

	return (fun_est, fun_loss, fun_step, fun_field)
end

# build functions for neural network
(fun_est, fun_loss, fun_step, fun_field) = generate_mt_hnn(n_in, ld)

function get_string(expr)
    replace(replace(replace(string(expr), r"#= [^*\s]* =#" => ""), r"\n[\s]*\n" => "\n"), "Num" => "")
end

# save function code to file -> mt_fun ... ModelingToolkit functions 
write("est.jl",   get_string(fun_est))
write("field.jl", get_string(fun_field))
write("loss.jl",  get_string(fun_loss))
write("step.jl",  get_string(fun_step))
