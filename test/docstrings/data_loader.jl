using Test
using GeometricMachineLearning
using GeometricMachineLearning: convert_input_and_batch_indices_to_array, number_of_batches

@testset "Data loader docstring examples" begin
    data = [1 2 3; 4 5 6]
    @test DataLoader(data; suppress_info = true).input == reshape(data, 2, 1, 3)

    tensor = [1; 2; 3;; 4; 5; 6;;; 7; 8; 9;; 10; 11; 12]
    expected_tensor = [1 4; 2 5; 3 6;;; 7 10; 8 11; 9 12]
    @test DataLoader(tensor; suppress_info = true).input == expected_tensor
    @test DataLoader(tensor; suppress_info = true) isa DataLoader

    dl = DataLoader(rand(5); suppress_info = true)
    batches = Batch(2)(dl)
    @test length(batches) == 3
    @test length.(batches) == (2, 2, 1)
    @test sort(collect(union(batches...))) == [(1, i) for i in 1:5]
    @test Batch(2, 3, 2) isa Batch{:Transformer}

    dat = [1, 2, 3, 4, 5]
    dl₁ = DataLoader(dat; autoencoder = false, suppress_info = true)
    dl₂ = DataLoader(dat; autoencoder = true, suppress_info = true)
    batch = Batch(3)
    batches₁ = batch(dl₁)
    batches₂ = batch(dl₂)
    @test number_of_batches(dl₁, batch) == number_of_batches(dl₂, batch) == 2
    @test length.(batches₁) == (3, 1)
    @test length.(batches₂) == (3, 2)
    @test sort(collect(union(batches₁...))) == [(i, 1) for i in 1:4]
    @test sort(collect(union(batches₂...))) == [(1, i) for i in 1:5]

    dl = DataLoader(collect(0.1:0.1:0.9); suppress_info = true)
    indices = [(1, 1), (1, 3), (1, 5)]
    @test convert_input_and_batch_indices_to_array(dl, Batch(3), indices) ==
        reshape([0.1, 0.3, 0.5], 1, 1, 3)
    @test number_of_batches(DataLoader([1, 2, 3, 4, 5]; suppress_info = true), Batch(2)) == 3
end
