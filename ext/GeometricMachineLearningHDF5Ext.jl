module GeometricMachineLearningHDF5Ext

using HDF5: HDF5
using Random: Xoshiro
using GeometricMachineLearning: LeftInverseSymplecticAutoencoder,
                                LeftInverseAutoencoderLayer,
                                NeuralNetwork, NetworkParameters, CPU
import GeometricMachineLearning: load

# `AbstractNeuralNetworks` loads a network as `NeuralNetwork(arch, Chain(arch), parameters, CPU())`. A
# `LeftInverseSymplecticAutoencoder` has no `Chain`: its model is one `LeftInverseAutoencoderLayer`,
# whose encoder and decoder share the parameter tree that `save` wrote.
function load(::Type{NeuralNetwork}, h5::HDF5.H5DataStore,
        arch::LeftInverseSymplecticAutoencoder, prototype...)
    model = LeftInverseAutoencoderLayer(arch)
    ps = load(NetworkParameters, h5, prototype...)
    # a fixed generator, so that a load leaves the caller's global one untouched
    _check_parameters(ps, NeuralNetwork(arch, model, CPU(), Float64; rng = Xoshiro(0)).params)
    NeuralNetwork(arch, model, ps, CPU())
end

# The file need not have been saved from `arch`: a biorthogonal network read into a Stiefel
# architecture, or the reverse, would load and fail only at its first call. Each layer must have the
# parameters of the layer `arch` initializes, and each of them its type and size; the element type
# is the file's.
function _check_parameters(ps::NetworkParameters, expected::NetworkParameters)
    keys(ps) == keys(expected) ||
        throw(ArgumentError("the file holds the layers $(keys(ps)), but the LeftInverseSymplecticAutoencoder has the layers $(keys(expected)); load the file with the architecture it was saved with."))
    for layer in keys(expected)
        Set(keys(ps[layer])) == Set(keys(expected[layer])) ||
            throw(ArgumentError("the layer `$layer` in the file has the parameters $(keys(ps[layer])), but the LeftInverseSymplecticAutoencoder expects $(keys(expected[layer])); load the file with the architecture it was saved with."))
        for name in keys(expected[layer])
            a, b = ps[layer][name], expected[layer][name]
            (nameof(typeof(a)) == nameof(typeof(b)) && size(a) == size(b)) ||
                throw(ArgumentError("the parameter `$layer.$name` in the file is a $(join(size(a), '×')) $(nameof(typeof(a))), but the LeftInverseSymplecticAutoencoder expects a $(join(size(b), '×')) $(nameof(typeof(b))); load the file with the architecture it was saved with."))
        end
    end
end

end
