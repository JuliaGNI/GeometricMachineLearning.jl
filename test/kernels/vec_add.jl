using CUDA 


function vec_add(x, v, w)
    i = threadIdx().x
    x[i] = v[i] + w[i]
    return 
end


vec_size = 10

v = CUDA.rand(vec_size)
w = CUDA.rand(vec_size)

x = CUDA.rand(vec_size)

@cuda threads=length(x) vec_add(x, v, w) 
