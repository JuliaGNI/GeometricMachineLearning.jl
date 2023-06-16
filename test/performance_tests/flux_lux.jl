# Import Packages
import Flux, Lux, Zygote, GeometricMachineLearning, Random, LinearAlgebra, Optimisers

# Generate data
data = [([x], 2x-x^3) for x in -2:0.1f0:2]

"""
This compares the performance between Lux and Flux

"""

########################################
# Training in a Flux framework

# Create Flux model 
function flux_training(n_steps)
  model = Flux.Chain(Flux.Dense(1 => 23, tanh), Flux.Dense(23 => 1, bias=false), only)

  # Create an optimiser
  optim = Flux.setup(Flux.Adam(), model)

  # Training 
  print("Optimization steps in Flux:\n")


  @time for epoch in 1:n_steps
    Flux.train!((m,x,y) -> (m(x) - y)^2, model, data, optim)
  end

end

####################################
#Training Lux with Optimisers.jl

function lux_optimisers_training(n_steps)

  model = Lux.Chain(Lux.Dense(1, 23, tanh), Lux.Dense(23, 1, bias=false))

  print("\n Optimization steps in Lux with Optimisers.jl: \n")

  # Initialize the Lux model
  ps, st = Lux.setup(Random.default_rng(), model)

  # set up optimizer
  st_opt = Optimisers.setup(Optimisers.Adam(0.001), ps)

  # Lost function
  loss(data, ps, st) = mapreduce(dat-> LinearAlgebra.norm(Lux.apply(model, dat[1], ps, st)[1] .- dat[2])^2,+,data)

  #training
  @time for epoch in 1:n_steps

    dp = Zygote.gradient(ps -> loss(data, ps, st), ps)[1]

    st_opt, ps = Optimisers.update(st_opt, ps, dp)

  end

end

########################################
# Training in a Lux framework

function lux_geometric_machine_learning_training(n_steps)
  # Create Lux model 
  model = Lux.Chain(Lux.Dense(1, 23, tanh), Lux.Dense(23, 1, bias=false))

  print("\n Optimization steps in Lux with GeometricMachineLearning.jl: \n")

  # Initialize the Lux model
  ps, st = Lux.setup(Random.default_rng(), model)

  #set up optimizer
  method = GeometricMachineLearning.AdamOptimizer()
  opt = GeometricMachineLearning.Optimizer(method, model)

  # Loss function
  loss(data, ps, st) = mapreduce(dat-> LinearAlgebra.norm(Lux.apply(model, dat[1], ps, st)[1] .- dat[2])^2,+,data)

  #training
  @time for epoch in 1:n_steps

    dp = Zygote.gradient(ps -> loss(data, ps, st), ps)[1]
    
    GeometricMachineLearning.optimization_step!(opt, model, ps, dp)
  end

end

for n_steps in 100:200:900
  print("Number of training steps is: ", n_steps, "\n")
  flux_training(n_steps)
  lux_optimisers_training(n_steps)
  lux_geometric_machine_learning_training(n_steps)
  print("\n")
end
