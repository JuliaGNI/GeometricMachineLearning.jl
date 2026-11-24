module GeometricMachineLearningHDF5Ext

using HDF5: HDF5
using GeometricMachineLearning: LeftInverseSymplecticAutoencoder,
                                LeftInverseAutoencoderLayer,
                                NeuralNetwork, NetworkParameters, CPU
import GeometricMachineLearning: load

# `AbstractNeuralNetworks` loads a network as `NeuralNetwork(arch, Chain(arch), parameters, CPU())`. A
# `LeftInverseSymplecticAutoencoder` has no `Chain`: its model is one `LeftInverseAutoencoderLayer`,
# whose encoder and decoder share the parameter tree that `save` wrote.
function load(::Type{NeuralNetwork}, h5::HDF5.H5DataStore,
        arch::LeftInverseSymplecticAutoencoder, prototype...)
    NeuralNetwork(arch, LeftInverseAutoencoderLayer(arch),
        load(NetworkParameters, h5, prototype...), CPU())
end

end
