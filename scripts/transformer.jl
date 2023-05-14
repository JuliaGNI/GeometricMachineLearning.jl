using GeometricMachineLearning, LinearAlgebra
import Lux, Zygote, Random 

#MNIST images are 28×28, so a sequence_length of 16 = 4² means the image patches are of size 7² = 49
image_dim = 28
patch_length = 7
patch_number = (image_dim÷patch_length)^2

train_x, train_y = MNIST(split=:train)[:]
test_x, test_y = MNIST(split=:test)[:]

#second argumen pl is "patch length"
#this splits the image into patches of size pl×pl and then arranges them into a matrix,
#the columns of the matrix give the patch number.
function split_image(image::AbstractMatrix, pl)
    n, m = size(image)
    @assert n == m
    @assert n%pl == 0
    #square root of patch number
    pnsq = n ÷ patch_length
    Tuple(vcat(map(j -> map(i -> image[pl*(i-1)+1:pl*i,pl*(j-1)+1:pl*j,1], 1:pnsq),1:pnsq)...))
end

function flatten(image_patch::AbstractMatrix)
    n, m = size(image_patch)
    reshape(image_patch, n*m)
end

function split_and_flatten(image::AbstractMatrix, pl)
    patch_number = (size(image, 1)÷pl)^2
    im_split = split_image(image, pl)
    hcat(Tuple(map(i -> flatten(im_split[i]), 1:patch_number))...)
end

#preprocessing steps 
train_x =   Tuple(map(i -> split_and_flatten(train_x[:,:,i], patch_length), 1:size(train_x,3)))
test_x =    Tuple(map(i -> split_and_flatten(test_x[:,:,i], patch_length), 1:size(test_x,3)))

train_y = Flux.onehotbatch(train_y, 0:9) #|> gpu
test_y = Flux.onehotbatch(test_y, 0:9) #|> gpu

#apply embedding before this!!

#encoder layer


Ψᵉ = Transformer(49, 7, Stiefel=true)

ps, st = Lux.setup(Random.default_rng(), Ψᵉ)  .|> gpu

#loss_sing
function loss_sing(ps, x, y)
    norm(Lux.apply(Ψᵉ, x, ps, st)[1] - y)
end
function loss_sing(ps, train_x, train_y, index)
    loss_sing(ps, train_x[:, index] |>gpu, train_y[:, index] |> gpu)    
end
function full_loss(ps, train_x, train_y)
    num = size(train_x, 2)
    mapreduce(index -> loss_sing(ps, train_x, train_y, index), +, 1:num)
end

num = size(train_x,2)
batch_size = 64
training_steps = 100


o = AdamOptimizer()
cache = init_optimizer_cache(Ψᵉ, o)

println("initial loss: ", full_loss(ps, train_x, train_y)/num)

@showprogress "Training network ..." for i in 1:training_steps
    index₁ = Int(ceil(rand()*num))
    x = train_x[:, index₁] |> gpu
    y = train_y[:, index₁] |> gpu
    l, pb = Zygote.pullback(ps -> loss_sing(ps, x, y), ps)
    dp = pb(one(l))[1]
    #dp = Zyogte.gradient(ps -> loss_sing(ps, x, y), ps)[1]

    indices = Int.(ceil.(rand(batch_size -1)*num))
    for index in indices
        x = train_x[:, index] |> gpu
        y = train_y[:, index] |> gpu
        l, pb = Zygote.pullback(ps -> loss_sing(ps, x, y), ps)
        dp = _add(dp, pb(one(l))[1])
    end
    optimization_step!(o, Ψᵉ, ps, cache, dp)
end
println("final loss: ", full_loss(ps, train_x, train_y)/num)

println("\nfinal test loss: ", full_loss(ps, test_x, test_y)/size(test_x, 2))