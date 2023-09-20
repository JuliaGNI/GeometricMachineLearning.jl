# Data Loader 

`GeometricMachineLearning` provides flexible routines to load and manage data for training neural networks. 
`DataLoader` has several constructors: 

1. If provided with a tensor, then it assumes the first axis is the system dimension, the second axis is the dimension of the parameter space, and the third axis gives the time evolution of the system. 

2. If provided with a tensor and a vector, it assumes the data are related to a classification task. 