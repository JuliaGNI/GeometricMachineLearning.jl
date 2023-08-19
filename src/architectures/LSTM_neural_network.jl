
struct LSTMNeuralNetwork{TS, TAO, TAM} <: Architecture
    dimin::Int
    dimst::Int
    size::TS
    σ₀::TAO
    σ₋₁::TAM
    function LSTMNeuralNetwork(dimin::Int, size::Tuple{<:Int, <:Int} = (1,1); dimst::Int = dimin, act0 = AbstractNeuralNetworks.SigmoidActivation(), act1 = tanh)
        @assert size[1] > 0 && size[2] > 0
        new{typeof(size), typeof(act0), typeof(act1)}(dimin, dimst, size, act0, act1)
    end
end

@inline AbstractNeuralNetworks.dim(arch::LSTMNeuralNetwork) = arch.dimin

function Chain(rnn::LSTMNeuralNetwork)
    N, M = rnn.size
    if N == 1
        cell_upper = reshape([LSTM(rnn.dimin, rnn.dimst, rnn.σ₀, rnn.σ₋₁) for _ in 1:M-1], (1,M-1))
        cell_output = [LSTM(rnn.dimin, rnn.dimst, rnn.σ₀, rnn.σ₋₁)]
        return GridCell([hcat(cell_upper, cell_output);])
    elseif M ==  1
        cell_upper = reshape([LSTM(rnn.dimin, rnn.dimst, rnn.σ₀, rnn.σ₋₁) for _ in 1:M], (1,M))
        cell_left  = [LSTM(rnn.dimst, rnn.dimst, rnn.σ₀, rnn.σ₋₁) for _ in 2:N-1]
        cell_bot   = reshape([LSTM(rnn.dimst, rnn.dimst, rnn.σ₀, rnn.σ₋₁) for _ in 1:M-1], (1,M-1))
        cell_output = [LSTM(rnn.dimst, rnn.dimst, rnn.σ₀, rnn.σ₋₁)]
        matrix = vcat(cell_upper, cell_left)
        matrix = vcat(matrix, hcat(cell_bot, cell_output))
        return GridCell(matrix)
    else
        cell_upper = reshape([LSTM(rnn.dimin, rnn.dimst, rnn.σ₀, rnn.σ₋₁) for _ in 1:M], (1,M))
        cell_left  = [LSTM(rnn.dimst, rnn.dimst, rnn.σ₀, rnn.σ₋₁) for _ in 2:N-1]
        cell_bot   = reshape([LSTM(rnn.dimst, rnn.dimst, rnn.σ₀, rnn.σ₋₁) for _ in 1:M-1], (1,M-1))
        cell_inner = [LSTM(rnn.dimst, rnn.dimst, rnn.σ₀, rnn.σ₋₁) for _ in 2:N-1, _ in 2:M]
        cell_output = [LSTM(rnn.dimst, rnn.dimst, rnn.σ₀, rnn.σ₋₁)]
        matrix = hcat(cell_left, cell_inner)
        matrix = vcat(cell_upper, matrix)
        matrix = vcat(matrix, hcat(cell_bot, cell_output))
        return GridCell(matrix)
    end
end
