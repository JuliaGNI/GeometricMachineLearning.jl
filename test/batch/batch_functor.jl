using Test
using GeometricMachineLearning
import Random 

Random.seed!(123)

const data_set_raw = 1:10
const data_set = reshape(data_set_raw, 1, 10)

const dl = DataLoader(data_set)

function test_batching_over_one_axis(batch_size::Int=3)
    batch = Batch(batch_size)
    tuple_indices = batch(dl)
    @test length(tuple_indices) == Int(ceil(10 / batch_size))
    tuple_indices_sorted = sort(vcat(tuple_indices...))
    indices_sorted = [tuple_index[2] for tuple_index in tuple_indices_sorted]
    @test indices_sorted == data_set_raw
end

test_batching_over_one_axis()
test_batching_over_one_axis(5)

const data_set_raw₂ = hcat(1:10, 21:30)
const data_set₂ = reshape(data_set_raw₂, 1, 10, 2)

const dl₂ = DataLoader(data_set₂)

function convert_vector_of_tuples_to_array(indices::Vector{<:Tuple})
    elem_in_tuple = length(indices[1])
    index_array = zeros(Int, length(indices), elem_in_tuple)
    for (i, elem) in zip(axes(indices, 1), indices)
        index_array[i, :] .= elem
    end
    index_array
end

sort_according_to_first_column(A::Matrix) = A[sortperm(A[:, 1]), :]

function test_batching_over_two_axis_with_seq_length(batch_size::Int=3, seq_length::Int=2, prediction_window::Int=1)
    batch = Batch(batch_size, seq_length, prediction_window)
    tuple_indices₂ = batch(dl₂)
    @test length(tuple_indices₂) == Int(ceil(2 * (10 - (seq_length - 1) - batch.prediction_window) / batch_size)) # have to multiply with two because we deal with tuples
    num_elems = 10 - (batch.seq_length -1) - batch.prediction_window
    indices₁ = hcat(1:num_elems, 1 * ones(Int, num_elems))
    indices₂ = hcat(1:num_elems, 2 * ones(Int, num_elems))
    true_indices₁ = sort_according_to_first_column(vcat(indices₁, indices₂))
    true_indices₂ = sort_according_to_first_column(vcat(indices₂, indices₁))
    # convert indices to a format such that we can compare them to the true indices
    indices_for_comparison = convert_vector_of_tuples_to_array(vcat(tuple_indices₂...))
    # sort the indices
    indices_for_comparison = sort_according_to_first_column(indices_for_comparison)
    @test indices_for_comparison == true_indices₁ || indices_for_comparison == true_indices₂
end

test_batching_over_two_axis_with_seq_length()
test_batching_over_two_axis_with_seq_length(3, 2, 2)