using GeometricMachineLearning
using GeometricMachineLearning: ForcingLayerP, ForcingLayerQP

forcing_layer_p = ForcingLayerP(2)
forcing_layer_qp = ForcingLayerQP(2)

nn_p = NeuralNetwork(forcing_layer_p)
nn_qp = NeuralNetwork(forcing_layer_qp)

println(parameterlength(nn_p))
println(parameterlength(nn_qp))