using Lux, Zygote, Random, Printf, LinearAlgebra

number_data_points = 100
data_input = 2*π*(1:number_data_points)/number_data_points
data_output = sin.(data_input)

data_input = Tuple([data_input[i]] for i in 1:number_data_points)
data_output = Tuple([data_output[i]] for i in 1:number_data_points)

layer_width = 100
model = Chain(Dense(1, layer_width, tanh), Dense(layer_width, 1))

ps, st = Lux.setup(Random.default_rng(), model)

loss(in, out, ps, st) = norm(Lux.apply(model, in, ps, st)[1] - out)

function gradient_eval(num = Int.(ceil.(rand()*number_data_points)))
    in = data_input[num]
    out = data_output[num]
    @printf "First one: "
    @time Zygote.gradient(ps -> loss(in, out, ps, st), ps)[1]
    @printf "Second one:"
    @time Zygote.gradient(ps -> loss(data_input[num], data_output[num], ps, st), ps)[1]
end

for i in 1:5
    gradient_eval()
end