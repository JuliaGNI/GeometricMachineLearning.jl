# Geodesic Retraction

General **retractions** are approximations of the exponential map. In `GeometricMachineLearning` we can, instead of using an approximation, solve the geodesic equation exactly (up to numerical error) by specifying `Geodesic()` as the argument of layers that have manifold weights. 