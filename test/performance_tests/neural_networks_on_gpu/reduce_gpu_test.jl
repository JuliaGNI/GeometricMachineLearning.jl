using CUDA 
using LinearAlgebra
using Printf

function cpu_execution(N, n_operations)
    A = rand(Float32, N, n_operations)
    @time a = reduce(max, A, dims=1)
    a
end

function gpu_execution(N, n_operations)
    A = CUDA.rand(N, n_operations)
    CUDA.@time a = reduce(max, A, dims=1)
    a 
end 

for N in 2 .^(12:15)
    for n_executions in 100:300:1000

        print("N = ",N," and number of executions is ", n_executions, "\n")

        @printf "execution on the cpu" 
        cpu_execution(N, n_executions)


        @printf "execution on the gpu" 
        gpu_execution(N, n_executions)

        print("\n")
    end
end