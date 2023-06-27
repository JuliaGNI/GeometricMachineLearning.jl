using Zygote, Printf, LinearAlgebra

number_data_points = 1000

data_input = Tuple([i] for i in 1:number_data_points)

function_to_be_differentiated(input, A) = norm(A*input)

function gradient_eval(num = Int(ceil(rand()*number_data_points)), A = rand(100000,1))
    input = data_input[num]
    @printf "First one: "
    @time Zygote.gradient(A -> function_to_be_differentiated(input, A), A)[1]
    @printf "Second one:"
    @time Zygote.gradient(A -> function_to_be_differentiated(data_input[num], A), A)[1]
    @printf "\n"
end

for i in 1:5
    gradient_eval()
end