using CUDA 

n_executions = 10
N = 64

# an expensive computation
function compute(A, B)
    C = A * B             # library call
    broadcast!(sin, C, C) # Julia kernel
    C
end

function gpu_run(A, B, n_executions)
    results = Vector{Any}(undef, n_executions)

    # computation
    @sync for i in 1:n_executions
        @async begin
            results[i] = Array(compute(A, B))
            nothing # JuliaLang/julia#40626
        end
    end
end

function main_gpu(N, n_executions)
    A = CUDA.rand(N,N)
    B = CUDA.rand(N,N)

    # make sure this data can be used by other tasks!
    #synchronize()

    gpu_run(A, B, n_executions)
end

function main_cpu(N, n_executions)
    results = Vector{Any}(undef, n_executions)

    A = rand(Float32, N, N)
    B = rand(Float32, N, N)
    
    for i in 1:n_executions
        results[i] = compute(A, B)
    end
end


for N in 2 .^(4:7)
    for n_executions in 10:30:100

        print("N = ",N," and number of executions is ", n_executions, "\n")

        @time "execution on the cpu" main_cpu(N, n_executions)


        @time "execution on the gpu" main_gpu(N, n_executions)

        print("\n")
    end
end