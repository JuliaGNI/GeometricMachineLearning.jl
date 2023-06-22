function vec_add!(x, v, w)
    i = threadIdx().x
    x[i] = v[i] + w[i]
    return 
end