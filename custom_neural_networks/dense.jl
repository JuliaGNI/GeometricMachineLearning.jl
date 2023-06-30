using CUDA 

struct FeedForwardLayer{use_bias, F1}
    n_in::Integer 
    n_out::Integer
    A::AbstractMatrix
    b::AbstractVector
    σ::F1
end

FeedForwardLayer(n_in::Integer, n_out::Integer, A::AbstractMatrix, b::AbstractVector, σ=tanh) = FeedForwardLayer{typeof(σ)}(n_in, n_out, A, b, σ)

FeedForwardLayer(n_in::Integer, n_out::Integer, σ=tanh) = FeedForwardLayer{typeof(σ)}(n_in, n_out, rand(n_out, n_in), rand(n_out), σ)

(layer::FeedForwardLayer)(x::AbstractMatrix) = layer.σ.(layer.Ax .+ layer.b)

n_in = 10
n_out = 2
n_data = 1000



layer = FeedForwardLayer(n_in, n_out, rand(n_out, n_in), rand(n_out))

X = rand(n_in, n_data)
