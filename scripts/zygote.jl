using Zygote, Printf, LinearAlgebra

const number_data_points = 1000

const data_input = [[i] for i in 1:number_data_points]

function_to_be_differentiated(input, A) = norm(A*input)

function gradient_eval(data, num, A = rand(100000,1))
    input = data[num]
    @printf "First one: "
    @time Zygote.gradient(A -> function_to_be_differentiated(input, A), A)[1]
    @printf "Second one:"
    @time Zygote.gradient(A -> function_to_be_differentiated(data[num], A), A)[1]
    @printf "\n"
end

for i in 1:5
    gradient_eval(data_input, Int(ceil(rand()*number_data_points)))
end
