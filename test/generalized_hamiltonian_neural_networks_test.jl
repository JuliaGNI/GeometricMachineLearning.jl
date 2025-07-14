using GeometricMachineLearning
using GeometricProblems.HarmonicOscillator: odeproblem, default_parameters
using GeometricIntegrators

sol = integrate(odeproblem(), ImplicitMidpoint())
dl = DataLoader(sol)

arch = ghnn(2)
nn = NeuralNetwork(arch)

