using Zygote, CUDA, Printf  

f(x) = sum(x.^5)

function compare(T, vector_dim)
    @printf "cpu"
    vec = rand(T, vector_dim)
    @time Zygote.gradient(f, vec)[1]

    @printf "gpu"
    vec_gpu = vec |> cu 
    CUDA.@time Zygote.gradient(f, vec_gpu)[1]
end

T = Float32
for lnvector_dim in 100:300:1000
    vector_dim = lnvector_dim^3
    @printf "vector_dim = %7i \n" vector_dim
    compare(T, vector_dim)
    @printf "\n"
end