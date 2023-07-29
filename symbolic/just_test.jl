using Symbolics


@variables x[1:1] y[1:2]

a = x[1]

z = a * y[1] * y[2]

Symbolics.gradient(z, y)





using GeometricMachineLearning
include("symbolic.jl")

hnn = NeuralNetwork(HamiltonianNeuralNetwork(2; nhidden = 0), Float64)

dimin = dim(hnn.architecture)



# creates variables for the input
@variables sinput[1:dimin]

# creates variables for the parameters
sparams = symbolicParams(hnn)

est = hnn(sinput, sparams)[1]



field =  Symbolics.gradient(est, sinput)

fun_est = build_function(est, sinput, develop(sparams)...)[2]

fun_field = build_function(field, sinput, develop(sparams)...)[1]

write("symbolic/field.jl", get_string(fun_field))

eval(fun_field)([1,2], develop(hnn.params)...)


