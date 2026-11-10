module GeometricMachineLearningHDF5Ext

using HDF5: HDF5
using GeometricMachineLearning: WeightTiedSymplecticAutoencoder, WeightTiedAutoencoderLayer, NeuralNetwork,
                                NetworkParameters, CPU
import GeometricMachineLearning: load

# `AbstractNeuralNetworks` loads a network as `NeuralNetwork(arch, Chain(arch), parameters, CPU())`. A
# `WeightTiedSymplecticAutoencoder` has no `Chain`: its model is one `WeightTiedAutoencoderLayer`, whose
# encoder and decoder share the parameter tree that `save` wrote.
function load(::Type{NeuralNetwork}, h5::HDF5.H5DataStore, arch::WeightTiedSymplecticAutoencoder, prototype...)
    NeuralNetwork(arch, WeightTiedAutoencoderLayer(arch), load(NetworkParameters, h5, prototype...), CPU())
end

end
