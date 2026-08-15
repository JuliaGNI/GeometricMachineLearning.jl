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

    @test onehotbatch([0]) == reshape([1, zeros(Int, 9)...], 10, 1, 1)
    input = [1 2 3 4 5 6; 7 8 9 10 11 12; 13 14 15 16 17 18;
             19 20 21 22 23 24; 25 26 27 28 29 30; 31 32 33 34 35 36]
    expected_patches = [1 19 4 22; 7 25 10 28; 13 31 16 34; 2 20 5 23;
                       8 26 11 29; 14 32 17 35; 3 21 6 24; 9 27 12 30;
                       15 33 18 36]
    @test split_and_flatten(input; patch_length = 3, number_of_patches = 4) == expected_patches

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
