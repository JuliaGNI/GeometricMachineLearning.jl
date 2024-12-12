@kernel function cpu_inverse_kernel!(B, A) 
    k = @index(Global)
    @views A_temp = A[:, :, k]
    @views B_temp = B[:, :, k]

    B_temp .= inv(A_temp)

    nothing
end

function cpu_inverse(A::AbstractArray)
    B = zero(A)
    backend = networkbackend(A)

    cpu_inverse! = cpu_inverse_kernel!(backend)
    cpu_inverse!(B, A, ndrange=size(A, 3))

    B 
end

@kernel function cpu_inverse_pullback_kernel!(dA, A, dB)
    k = @index(Global)
    @views A_temp = A[:, :, k]
    @views dA_temp = dA[:, :, k]
    @views dB_temp = dB[:, :, k]

    copy!(dA_temp, Zygote.pullback(inv, A_temp)[2](dB_temp)[1])

    nothing
end

function ChainRulesCore.rrule(::typeof(cpu_inverse), A::AbstractArray)
    B = cpu_inverse(A)

    function cpu_inverse_pullback(dB::AbstractArray)
        dA = zero(dB)
        backend = networkbackend(dB)

        cpu_inverse_pullback! = cpu_inverse_pullback_kernel!(backend)
        cpu_inverse_pullback!(dA, A, dB, ndrange=size(dB, 3))

        return NoTangent(), dA
    end

    return B, cpu_inverse_pullback
end

function cpu_tensor_cayley(A::AbstractArray)
    one_A = init_output(A)

    tensor_tensor_mul(one_A - A, cpu_inverse(one_A + A))
end