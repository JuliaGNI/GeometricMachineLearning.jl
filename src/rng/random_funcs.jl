#Lux is always working with single precision!
function Lux.glorot_uniform(rng::TrivialInitRNG, dims::Integer...; gain = 1)
    rand(rng, Float32, dims...)
end
 
function Lux.glorot_uniform(rng::TrivialInitRNG, T::Type{StiefelManifold}, N, n; gain = 1)
    rand(rng, T, N, n)
end

function Lux.glorot_uniform(rng::TrivialInitRNG, T::Type{SymplecticStiefelManifold}, N, n; gain = 1)
    rand(rng, T, N, n)
end

function Lux.glorot_uniform(rng::AbstractRNG, ::Type{StiefelManifold}, N, n; gain = 1)
    A = Lux.glorot_uniform(rng, N, n; gain=gain)
    StiefelManifold(qr!(A).Q[1:N, 1:n])
end

function Lux.glorot_uniform(rng::AbstractRNG, ::Type{SymplecticStiefelManifold}, N2, n2; gain = 1)
    A = Lux.glorot_uniform(rng, N2, n2; gain=gain)
    SymplecticStiefelManifold(sr!(A).S[1:N2, vcat(1:(n2÷2), (n2÷2 + 1):n2)])
end