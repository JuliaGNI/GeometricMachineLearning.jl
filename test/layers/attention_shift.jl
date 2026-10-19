# `SymplecticAttention` and `LinearSymplecticAttention` update one half of the input through
# `_shift(Val(C), z, f)`, one method per weighting. The outputs below are pinned with `==` against
# an independent formula written as the same operations: the pin is bit for bit, and a swap of
# `_shift`'s two branches fails every one of them.
#
# `SymplecticAttention` weights the scalar products with a `SymmetricMatrix` (`symmetric = true`) or
# a dense matrix (`symmetric = false`), so the two leaves of the part are both here. The activation
# is `MatrixSoftmax`, the layer's default.

using GeometricMachineLearning
using Test

# Sequence data: dimension 4 halves to two rows, and three positions per sequence.
function sequence_data(::Type{T}) where {T}
    q = T[0.25 1.5 -0.5; -0.75 0.5 2.0]
    p = T[1.0 -0.25 0.75; -0.5 1.25 -1.0]
    (q = q, p = p)
end

const DENSE_WEIGHT = [1.0 -2.0; 0.5 3.0]

@testset "SymplecticAttention pins the half it changes ($T)" for T in (Float32, Float64)
    z = sequence_data(T)
    σ = MatrixSoftmax()

    @testset "the symmetric weighting, a SymmetricMatrix leaf" begin
        A = SymmetricMatrix(T[1, 2, 3], 2)
        ps = (A = A,)
        G = Matrix(A)

        for d in (SymplecticAttentionQ(4), SymplecticAttentionP(4))
            half = d isa SymplecticAttentionQ ? z.p : z.q
            update = (G * half) * (2 * σ((half' * G) * half))
            y = d(z, ps)
            if d isa SymplecticAttentionQ
                @test y.q == z.q + update
                @test y.p == z.p
            else
                @test y.q == z.q
                @test y.p == z.p + update
            end
        end

        @test typeof(ps.A) <: SymmetricMatrix
    end

    @testset "the arbitrary weighting, a dense leaf" begin
        A = T.(DENSE_WEIGHT)
        ps = (A = A,)

        for d in (SymplecticAttentionQ(4; symmetric = false),
            SymplecticAttentionP(4; symmetric = false))
            half = d isa SymplecticAttentionQ ? z.p : z.q
            s = σ((half' * A) * half)
            update = (A * half) * s' + (A' * half) * s
            y = d(z, ps)
            if d isa SymplecticAttentionQ
                @test y.q == z.q + update
                @test y.p == z.p
            else
                # The layer adds the two products before the shift, not after.
                @test y.p == z.p + update
                @test y.q == z.q
            end
        end
    end

    @testset "a 3-tensor input, against the reference to rounding" begin
        # A batch of sequences. The tensor path of `_mul` is one GEMM over the slices where the
        # reference below multiplies slice by slice, so the two agree to rounding here and not bit
        # for bit; the matrix case above carries the exact pin.
        A = SymmetricMatrix(T[1, 2, 3], 2)
        G = Matrix(A)
        X = (q = repeat(z.q, 1, 1, 2), p = repeat(z.p, 1, 1, 2))
        for d in (SymplecticAttentionQ(4), SymplecticAttentionP(4))
            Y = d(X, (A = A,))
            reference = similar(X.q)
            half = d isa SymplecticAttentionQ ? X.p : X.q
            for k in axes(X.q, 3)
                reference[:, :, k] = (G * half[:, :, k]) *
                                     (2 * σ((half[:, :, k]' * G) * half[:, :, k]))
            end
            half_out = d isa SymplecticAttentionQ ? Y.q : Y.p
            @test isapprox(half_out, (d isa SymplecticAttentionQ ? X.q : X.p) + reference;
                rtol = 8 * eps(T))
        end
    end
end

@testset "LinearSymplecticAttention pins the half it changes ($T)" for T in (Float32, Float64)
    z = (q = T[0.25 1.5 -0.5; -0.75 0.5 2.0; 0.5 0.25 -1.0; 1.0 -0.5 0.75],
        p = T[1.0 -0.25 0.75; -0.5 1.25 -1.0; 0.25 0.5 -0.75; -1.0 0.75 0.5])
    # The weight is a `seq_length × seq_length` symmetric matrix; the layer multiplies it from the
    # right.
    A = SymmetricMatrix(T[1, 2, 3, 4, 5, 6], 3)
    G = Matrix(A)
    ps = (A = A,)

    for (d, expected) in ((LinearSymplecticAttentionQ(4, 3),
        (q = z.q + z.p * G, p = z.p)),
        (LinearSymplecticAttentionP(4, 3), (q = z.q, p = z.p + z.q * G)))
        y = d(z, ps)
        @test y.q == expected.q
        @test y.p == expected.p
    end
end

@testset "one method per weighting, and the two halves dispatch to it" begin
    # A `Q` layer and its `P` counterpart pass a different `Val` to the same method. The argument
    # types are the layer's own `(q, p)` `NamedTuple`, which its signature names exactly.
    arguments = (typeof((q = zeros(2, 3), p = zeros(2, 3))), NamedTuple)

    @test which(SymplecticAttentionQ(4), arguments) ===
          which(SymplecticAttentionP(4), arguments)
    @test which(SymplecticAttentionQ(4; symmetric = false), arguments) ===
          which(SymplecticAttentionP(4; symmetric = false), arguments)

    # The two weightings are still two methods: the symmetric path reads the one weight, and the
    # arbitrary path reads its transpose as well.
    @test which(SymplecticAttentionQ(4), arguments) !==
          which(SymplecticAttentionQ(4; symmetric = false), arguments)

    attention_arguments = (typeof((q = zeros(4, 3), p = zeros(4, 3))), NamedTuple)
    @test which(LinearSymplecticAttentionQ(4, 3), attention_arguments) ===
          which(LinearSymplecticAttentionP(4, 3), attention_arguments)
end
