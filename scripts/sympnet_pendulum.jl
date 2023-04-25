"""
Simple implementation of a pendulum for SympNets.
"""

using GeometricIntegrators
using Plots
using GeometricMachineLearning
using Lux
using Zygote
using LinearAlgebra
import Random 

#Hamiltonian & vector fields 
H(t, q, p, params) = .5*p[1]^2 + (1-cos(q[1]))
function v(v, t, q, p, params)
    v[1] = p[1]
end
function f(f, t, q, p, params)
    f[1] = -sin(q[1])
end

#simulate data with geometric Integrators
T = 100; Δt = 0.1; 
ode = HODEProblem(v, f, H, (0, T), Δt, (q=randn(1), p=randn(1)))
int = Integrator(ode, SymplecticTableau(TableauExplicitEuler()))
sol = integrate(ode, int)
q = zeros(length(sol.q)); p = copy(q)
for i in 0:(length(sol.q)-1)
    q[i+1] = sol.q[i][1]
    p[i+1] = sol.p[i][1]
end 
p1 = plot(q, p, label="Training data.")

#Sympnet model 
model = Chain(  Gradient(2, 10, tanh), 
                Gradient(2, 10, tanh; change_q=false)
                #Gradient(2, 10, tanh),
                #Gradient(2, 10, tanh; change_q=false)
                )

ps, st = Lux.setup(Random.default_rng(), model)

#defines a loss function 
function loss(ps, batch_size=10)
    loss = 0 
    for i in 1:batch_size
        index = Int(ceil(rand()*T/Δt))
        qp_new = Lux.apply(model, [q[index], p[index]], ps, st)[1]
        loss += norm(qp_new - [q[index+1], p[index+1]])
    end
    loss 
end

function full_loss(ps)
    loss = 0 
    for i in 1:Int(T/Δt)
        qp_new = Lux.apply(model, [q[i], p[i]], ps, st)[1]
        loss += norm(qp_new - [q[i+1], p[i+1]])
    end
    loss 
end 

#define momentum optimizer and initialize
o = MomentumOptimizer(1e-2, 0.5)
#initial gradients for calling Cache constructor
dp_in = Zygote.gradient(ps -> loss(ps), ps)[1]
cache = MomentumOptimizerCache(o, model, ps, dp_in)

#training 
print("loss is: ", full_loss(ps), "\n")
training_steps = 1000
for i in 1:training_steps
    dp = Zygote.gradient(ps -> loss(ps), ps)[1]
    apply!(o, cache, model, ps, dp)
end 
print("loss is: ", full_loss(ps))

#evaluate pendulum trajectory for the inital conditions for which it was trained
q_learned = zeros(length(sol.q))
p_learned = zeros(length(sol.p))
q_learned[1] = q[1]
p_learned[1] = p[1]

for i in 2:length(sol.q)
    q_learned[i], p_learned[i] = Lux.apply(model, [q_learned[i-1], p_learned[i-1]], ps, st)[1]
end 
plot(p1, q_learned, p_learned, label="Learned trajectory.")