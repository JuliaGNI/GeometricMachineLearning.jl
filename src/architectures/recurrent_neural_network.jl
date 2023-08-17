
struct RecurrentNeuralNetwork{TS, TAO, TAM} <: Architecture
    dimin::Int
    dimout::Int
    dimst::Int
    size::TS
    act_output::TAO
    act_st::TAM
    function RecurrentNeuralNetwork(dimin::Int, dimout::Int, size::Tuple{<:Int, <:Int} = (1,1); dimst::Int = dimin, act_out = tanh, act_st =act_out)
        @assert size[1] > 0 && size[2] > 0
        new{typeof(size), typeof(act_out), typeof(act_st)}(dimin, dimout, dimst, size, act_out, act_st)
    end
end

@inline AbstractNeuralNetworks.dim(arch::RecurrentNeuralNetwork) = arch.dimin

function Chain(rnn::RecurrentNeuralNetwork)
    N, M = rnn.size
    if N == 1
        cell_upper = reshape([Recurrent(rnn.dimin, rnn.dimst, 0, rnn.dimst, rnn.act_output, rnn.act_st) for _ in 1:M-1], (1,M-1))
        cell_output = [Recurrent(rnn.dimin, rnn.dimst, rnn.dimout, rnn.dimst, rnn.act_output, rnn.act_st)]
        return GridCell([hcat(cell_upper, cell_output);])
    elseif M ==  1
        cell_upper = reshape([Recurrent(rnn.dimin, rnn.dimst,rnn.dimst, rnn.dimst, rnn.act_output, rnn.act_st) for _ in 1:M], (1,M))
        cell_left  = [Recurrent(rnn.dimst, rnn.dimst, rnn.dimst, rnn.dimst, rnn.act_output, rnn.act_st) for _ in 2:N-1]
        cell_bot   = reshape([Recurrent(rnn.dimst, rnn.dimst,0, rnn.dimst, rnn.act_output, rnn.act_st) for _ in 1:M-1], (1,M-1))
        cell_output = [Recurrent(rnn.dimin, rnn.dimst, rnn.dimout, rnn.dimst, rnn.act_output, rnn.act_st)]
        matrix = vcat(cell_upper, cell_left)
        matrix = vcat(matrix, hcat(cell_bot, cell_output))
        return GridCell(matrix)
    else
        cell_upper = reshape([Recurrent(rnn.dimin, rnn.dimst,rnn.dimst, rnn.dimst, rnn.act_output, rnn.act_st) for _ in 1:M], (1,M))
        cell_left  = [Recurrent(rnn.dimst, rnn.dimst, rnn.dimst, rnn.dimst, rnn.act_output, rnn.act_st) for _ in 2:N-1]
        cell_bot   = reshape([Recurrent(rnn.dimst, rnn.dimst, 0, rnn.dimst, rnn.act_output, rnn.act_st) for _ in 1:M-1], (1,M-1))
        cell_inner = [Recurrent(rnn.dimst, rnn.dimst, rnn.dimst, rnn.dimst, rnn.act_output, rnn.act_st) for _ in 2:N-1, _ in 2:M]
        cell_output = [Recurrent(rnn.dimin, rnn.dimst, rnn.dimout, rnn.dimst, rnn.act_output, rnn.act_st)]
        matrix = hcat(cell_left, cell_inner)
        matrix = vcat(cell_upper, matrix)
        matrix = vcat(matrix, hcat(cell_bot, cell_output))
        return GridCell(matrix)
    end
end
