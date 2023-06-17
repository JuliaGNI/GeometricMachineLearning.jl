# Import Packages
import Flux, Lux, Zygote, GeometricMachineLearning, Random, LinearAlgebra

# Generate data
data = [([x], 2x-x^3) for x in -2:0.1f0:2]

"""
This compares the performance between Lux and Flux

"""

########################################
# Training in a Flux framework

# Create Flux model 
model = Flux.Chain(Flux.Dense(1 => 23, tanh), Flux.Dense(23 => 1, bias=false), only)

# Create an optimiser
optim = Flux.setup(Flux.Adam(), model)


# Training 
print("Optimization steps in Flux:\n")

@time for epoch in 1:1000
  Flux.train!((m,x,y) -> (m(x) - y)^2, model, data, optim)
end


########################################
# Training in a Lux framework

# Create Lux model 
model = Lux.Chain(Lux.Dense(1, 23, tanh), Lux.Dense(23, 1, bias=false))

# Initialize the Lux model
ps, st = Lux.setup(Random.default_rng(), model)

#creation of optimiser
method = GeometricMachineLearning.AdamOptimizer()
opt = GeometricMachineLearning.Optimizer(method, model)

# Training 
print("\n Optimization steps in Lux: \n")

# Lost function
loss(data, ps, st) = mapreduce(dat-> LinearAlgebra.norm(Lux.apply(model, dat[1], ps, st)[1] .- dat[2])^2,+,data)

@time for epoch in 1:1000

    dp = Zygote.gradient(ps -> loss(data, ps, st), ps)
    
    GeometricMachineLearning.optimization_step!(opt, model, ps, dp[1])

end