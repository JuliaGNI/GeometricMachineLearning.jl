using Zygote
using Symbolics


develop(x) = [x]
develop(t::Tuple{Any}) = [develop(t[1])...]
develop(t::Tuple) = [develop(t[1])..., develop(t[2:end])...]
function develop(t::NamedTuple) 
   X = [[develop(e)...] for e in t] 
   vcat(X...)
end

@variables SX[1:2]
@variables W1[1:2,1:2] W2[1:2,1:2] b1[1:2] b2[1:2]
SV = ((W = W1, b = b1), (W = W2, b = b2))

@variables SSV::typeof(SV)


z = SV[2].W  * tanh.(SV[1].W * SX + SV[1].b) + SV[2].b

fun = build_function(z, SX, develop(SV)...)[2]
f = eval(fun)

V = ((W = [1 3; 2 2], b = [1, 0]), (W = [1 1; 0 2], b = [1, 0]))
X = [1, 0.2]

f(X, develop(V)...) 

#Calcul du gradient

Base.size(nt::NamedTuple) = (length(nt),)
(::Zygote.ProjectTo{AbstractArray})(x::Tuple{Vararg{Any}})  = [x...]

df(x, v) = Zygote.gradient(x->sum(f(x, develop(v)...)), x)[1]

df(X, V)

loss(x,v) = Zygote.gradient(p->sum(df(x,p)), v)

loss(X, V)


#= Test sans develop

@variables SY[1:2]
@variables W1[1:2,1:2] W2[1:2,1:2] 
SW = (W = W1, b = b1)

z2 = tanh.(SW.W * SY + SW.b)

fun2 = build_function(z2, SY, SW...)[2]
f2 = eval(fun2)

W = (W = [1 3; 2 2], b = [1, 0])
Y = [1, 0.2]

f2(Y, W...) 

df2(y, w) = Zygote.gradient(y->sum(f2(y, w...)), y)[1]

df2(Y, W)

loss2(y,w) = Zygote.gradient(p->sum(df2(y,p)), w)

loss2(Y,W)

# test sans develop NamedTuple

@variables SY2[1:2]
@variables W1[1:2,1:2] W2[1:2,1:2] 
SW2 = ((W1,), (b1,))

z3 = tanh.(SW2[1][1] * SY2 + SW2[2][1])

fun3 = build_function(z3, SY2, develop(SW2)...)[2]
f3 = eval(fun3)

W2 = (([1 3; 2 2],),([1, 0],))
Y2 = [1, 0.2]

f3(Y2, develop(W2)...) 

df3(y, w) = Zygote.gradient(y->sum(f3(y, develop(w)...)), y)[1]

df3(Y2, W2)

loss3(y,w) = Zygote.gradient(p->sum(df3(y,p)), w)

loss3(Y2,W2)
=#