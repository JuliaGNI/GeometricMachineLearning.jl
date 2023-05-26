#Lux is always working with single precision!
function Lux.glorot_uniform(rng::TrivialInitRNG, dims::Integer...; gain = 1)
    rand(rng, Float32, dims...)
end
 
function Lux.glorot_uniform(rng::TrivialInitRNG, ::Type{StiefelManifold}, N, n; gain = 1)
    rand(rng, StiefelManifold{Float32}, N, n)
end

function Lux.glorot_uniform(rng::AbstractRNG, ::Type{StiefelManifold}, N, n; gain = 1)
    A = Lux.glorot_uniform(rng, N, n; gain=gain)
    StiefelManifold(qr(A).Q[1:N, 1:n])
end