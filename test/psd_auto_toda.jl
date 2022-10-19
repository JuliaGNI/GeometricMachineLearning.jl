using Plots
using LinearAlgebra
using Zygote 
using NLsolve

import ForwardDiff

include("misc.jl")

N = 10

c = .1
l = 1
Δx = l/N

J_N = make_J(N)
link_matrix  = zeros(N,N)
link_matrix[1,N] = 1
for i in 2:N link_matrix[i,i-1] = 1 end

toda_pot = r -> exp(-r) + r - 1

function H(z,c,l,Δx,link_matrix)
        N = length(z)÷2
        #momentum terms
        H_out = .5*Δx*sum(z[N+1:2*N].^2)
        #potential
        q = z[1:N]
        Q = repeat(q', outer=(N,1))
        #quadratic distances
        CQ = toda_pot.(Q - Q')
        H_out = H_out + .5*c^2/Δx*tr(link_matrix*CQ)
        return H_out
end


function Xₕ(z,c,l,Δx,link_matrix,J_N)
    return J_N*gradient(ζ -> H(ζ,c,l,Δx,link_matrix),z)[1]
end


n_steps = Int(5e4)
step_size = 0.05
t = step_size*(0:n_steps)

#set initial conditions
s(x) = 10*abs(x-.5)
function h(s)
    if 0≤s≤1 
        return 1. - 3/2*s^2 + 3/4*s^3
    elseif 1<s≤2
        return 1/4*(2. -s)^3
    elseif s>2
        return zeros(length(s))
    else
        throw(DomainError(ch),"s must be ≥0!")
    end
end

x_vec = l/N:l/N:l
z_init = zeros(2*N)
for i in 1:N z_init[i] = h(s(x_vec[i]))[1] end
z_vec = zeros(2*N,n_steps+1)
z_vec[:,1] = z_init

for it in 1:n_steps
        f(z) = z - z_vec[:,it] - step_size*Xₕ(.5*(z+z_vec[:,it]),c,l,Δx,link_matrix,J_N)
        z_vec[:,it+1] = nlsolve(f,z_vec[:,it],autodiff=:forward).zero
end

data₁ = z_vec
data₂ = hcat(z_vec[1:N,:],z_vec[N+1:2*N])

#number of modes
n_m = 3

U₁ = svd(data₁).U[:,1:2*n_m]
U₂ = svd(data₂).U[:,1:n_m]

Φ = vcat(hcat(U₂,zeros(N,n_m)),hcat(zeros(N,n_m),U₂))

function X_red(ξ,c,l,Δx,link_matrix,Ψ,J_N)
    return Ψ'*J_N*gradient(ζ -> H(ζ,c,l,Δx,link_matrix),Ψ*ξ)[1]
end

ξ_init₁ = U₁'*z_init
ξ_init₂ = Φ'*z_init
ξ_vec₁ = zeros(2*n_m,n_steps+1)
ξ_vec₂ = zeros(2*n_m,n_steps+1)
ξ_vec₁[:,1] = ξ_init₁
ξ_vec₂[:,1] = ξ_init₂

for it in 1:n_steps
        f₁(ξ) = ξ - ξ_vec₁[:,it] - step_size*X_red(.5*(ξ+ξ_vec₁[:,it]),c,l,Δx,link_matrix,U₁,J_N)
        ξ_vec₁[:,it+1] = nlsolve(f₁,ξ_vec₁[:,it],autodiff=:forward).zero
        f₂(ξ) = ξ - ξ_vec₂[:,it] - step_size*X_red(.5*(ξ+ξ_vec₂[:,it]),c,l,Δx,link_matrix,Φ,J_N)
        ξ_vec₂[:,it+1] = nlsolve(f₂,ξ_vec₂[:,it],autodiff=:forward).zero
end


err₁ = .5*(sum((z_vec - U₁*ξ_vec₁).^2,dims=1).^.5)
err₂ = .5*(sum((z_vec - Φ*ξ_vec₂).^2,dims=1).^.5)
abs_val = .5*(sum(z_vec.^2,dims=1).^.5)

p₁ = plot(vec(err₁./abs_val))
p₂ = plot(vec(err₂./abs_val))

#neural network part
nn_in = Chain(Gradient(2*N,4*N),Gradient(2*N,4*N;change_q=false),SymplecticStiefelLayer(2*n_m,2*N;inverse=true),Gradient(2*n_m,4*n_m))
nn_out = Chain(Gradient(2*n_m,4*n_m),SymplecticStiefelLayer(2*n_m,2*N),Gradient(2*N,4*N;change_q=false),Gradient(2*N,4*N))
nn_total = Chain(nn_in,nn_out)

ps, st = Lux.setup(Random.default_rng(), nn_total)

o = StandardOptimizer(1f-5)
g = gradient(p -> norm(z_vec - Lux.apply(nn_total, z_vec, p, st)[1]), ps)
apply!(o, ps, g, st)

ps_in = (layer_1=ps[1],layer_2=ps[2],layer_3=ps[3])
st_in = (layer_1=st[1],layer_2=st[2],layer_3=st[3])


function X_nn(ξ,ps_in,st_in,c,l,Δx,link_matrix,J_n)
    return J_n*gradient(ξ₁ -> H(Lux.apply(nn_in,ξ₁, ps_in, st_in)[1],c,l,Δx,link_matrix),ξ)[1]
end

J_n = make_J(n_m)

ξ_init_nn = Π(z_init,Aa,X)
ξ_vec_nn = zeros(2*n_m,n_steps+1)
ξ_vec_nn[:,1] = ξ_init_nn

for it in 1:n_steps
        f(ξ) = ξ - ξ_vec_nn[:,it] - step_size*X_nn(.5*(ξ+ξ_vec_nn[:,it]),ps_in,st_in,c,l,Δx,link_matrix,J_n)
        ξ_vec_nn[:,it+1] = nlsolve(f,ξ_vec_nn[:,it],autodiff=:finite).zero
end


#err_nn = .5*(sum((z_vec-Ξ(ξ_vec_nn,Aa,X)).^2,dims=1).^.5)
#p_nn = plot(vec(err_nn./abs_val))

#anim1 = @animate for 1∈1:100:n_steps
#    plot((U₁*ξ_vec₁)[1:N,i],yrange=[0,1])
#end
#gif(anim1,"toda_pod",fps=10)





