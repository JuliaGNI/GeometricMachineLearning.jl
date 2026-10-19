# The three SympNet layer families update one half of the input, through `_shift(Val(C), z, f)`.
# Every output below is asserted with `==` against an independent formula written as the same
# operations: the pin is bit for bit. The parameters and the input are written out rather than
# drawn, so the pin is a property of the layer and not of an initialiser's draw.
#
# Swapping `_shift`'s two branches moves the change into the other half of the tuple, which fails
# every assertion here — both the half that should have moved and the half that should not.

using GeometricMachineLearning
using GeometricMachineLearning: _shift
using Test

# Dimension 4, upscaling dimension 6: the upscaling dimension halves to three rows, so the weight is
# `3 × 2` and the `SymmetricMatrix` holds the lower triangle of a `2 × 2` block.
function parameters(::Type{T}) where {T}
    (weight = T[1 2; 3 -1; 0.5 4], bias = T[0.1, -0.2, 0.3], scale = T[0.5, -1.0, 2.0],
        symmetric = SymmetricMatrix(T[1, 2, 3], 2))
end

@testset "the SympNet layers pin the half they change ($T)" for T in (Float32, Float64)
    p = parameters(T)
    q = T[0.25, -0.75]
    p_half = T[1.0, -0.5]

    @testset "GradientLayer" begin
        ps = (weight = p.weight, bias = p.bias, scale = p.scale)
        x = (q = q, p = p_half)
        on_p = p.weight' * (p.scale .* tanh.(p.weight * p_half .+ p.bias))
        on_q = p.weight' * (p.scale .* tanh.(p.weight * q .+ p.bias))

        for (d, expected) in ((GradientLayerQ(4, 6, tanh), (q = q + on_p, p = p_half)),
            (GradientLayerP(4, 6, tanh), (q = q, p = p_half + on_q)))
            y = d(x, ps)
            @test y.q == expected.q
            @test y.p == expected.p
            @test eltype(y.q) === T
            @test eltype(y.p) === T
        end

        # The matrix argument of the same layer: the bias and the scale broadcast down the columns.
        X = (q = repeat(q, 1, 3), p = repeat(p_half, 1, 3))
        for (d, half) in ((GradientLayerQ(4, 6, tanh), p_half), (
            GradientLayerP(4, 6, tanh), q))
            Y = d(X, ps)
            moved = p.weight' * (p.scale .* tanh.(p.weight * repeat(half, 1, 3) .+ p.bias))
            @test Y.q == (d isa GradientLayerQ ? X.q + moved : X.q)
            @test Y.p == (d isa GradientLayerQ ? X.p : X.p + moved)
            @test eltype(Y.q) === T
            @test eltype(Y.p) === T
        end
    end

    @testset "ActivationLayer" begin
        ps = (scale = p.scale[1:2],)
        x = (q = q, p = p_half)

        for (d, expected) in ((
            ActivationLayerQ(4, tanh), (q = q + ps.scale .* tanh.(p_half),
                p = p_half)),
            (ActivationLayerP(4, tanh), (q = q, p = p_half + ps.scale .* tanh.(q))))
            y = d(x, ps)
            @test y.q == expected.q
            @test y.p == expected.p
            @test eltype(y.q) === T
            @test eltype(y.p) === T
        end

        X = (q = repeat(q, 1, 3), p = repeat(p_half, 1, 3))
        @test ActivationLayerQ(4, tanh)(X, ps).q == X.q .+ ps.scale .* tanh.(X.p)
        @test ActivationLayerP(4, tanh)(X, ps).p == X.p .+ ps.scale .* tanh.(X.q)
        @test eltype(ActivationLayerQ(4, tanh)(X, ps).q) === T
        @test eltype(ActivationLayerP(4, tanh)(X, ps).p) === T
    end

    @testset "LinearLayer" begin
        ps = (weight = p.symmetric,)
        x = (q = q, p = p_half)

        for (d, expected) in ((LinearLayerQ(4), (q = q + p.symmetric * p_half, p = p_half)),
            (LinearLayerP(4), (q = q, p = p_half + p.symmetric * q)))
            y = d(x, ps)
            @test y.q == expected.q
            @test y.p == expected.p
            @test eltype(y.q) === T
            @test eltype(y.p) === T
        end

        X = (q = repeat(q, 1, 3), p = repeat(p_half, 1, 3))
        @test LinearLayerQ(4)(X, ps).q == X.q .+ p.symmetric * X.p
        @test LinearLayerP(4)(X, ps).p == X.p .+ p.symmetric * X.q
        @test LinearLayerQ(4)(X, ps).p == X.p
        @test LinearLayerP(4)(X, ps).q == X.q
        @test eltype(LinearLayerQ(4)(X, ps).q) === T
        @test eltype(LinearLayerP(4)(X, ps).p) === T
    end

    @testset "a half of the wrong length is rejected" begin
        # Dimension 4 acts on halves of two rows. The parameters below are shaped to a half of
        # three, so nothing but the layer's own check rejects the input: a one-element scale
        # broadcasts, and the 3×3 and 4×3 weights multiply the three rows it is given.
        wrong = (q = T[1, 2, 3], p = T[1, 2, 3])
        weight = T[1 0.5 -1; 0 1 0.5; -1 0 1]
        gradient = (weight = T[1 0 0.5; 0 1 -1; 0.5 0 1; 1 -1 0],
            bias = T[0.1, -0.2, 0.3, 0.4], scale = T[0.5, -1.0, 2.0, 1.0])
        cases = ((ActivationLayerQ(4, tanh), (scale = T[0.5],)),
            (ActivationLayerP(4, tanh), (scale = T[0.5],)),
            (GradientLayerQ(4, 6, tanh), gradient),
            (GradientLayerP(4, 6, tanh), gradient),
            (LinearLayerQ(4), (weight = weight,)),
            (LinearLayerP(4), (weight = weight,)))
        for (d, ps) in cases
            @test_throws "Dimension mismatch." d(wrong, ps)

            # The array path splits six rows into halves of three and reaches the same check.
            @test_throws "Dimension mismatch." d(zeros(T, 6), ps)

            # The check reads `q`: a pair whose halves differ in length is rejected as well, so a
            # `p` of the layer's own length cannot stand in for the dimension.
            @test_throws "Dimension mismatch." d((q = T[1, 2, 3], p = T[1, 2]), ps)
        end
    end
end

@testset "each family is one method, and its two halves dispatch to it" begin
    # `_shift` is the one place the two halves are told apart.
    @test length(methods(_shift)) == 2

    # A `Q` layer and its `P` counterpart pass a different `Val` to the same functor; a method per
    # half would give two distinct methods here.
    for (q_layer, p_layer) in ((GradientLayerQ(4, 6, tanh), GradientLayerP(4, 6, tanh)),
        (ActivationLayerQ(4, tanh), ActivationLayerP(4, tanh)),
        (LinearLayerQ(4), LinearLayerP(4)))
        @test which(q_layer, (NamedTuple, NamedTuple)) ===
              which(p_layer, (NamedTuple, NamedTuple))
    end
end

@testset "the dimension field is concrete, and an Int32 dimension is stored as an Int" begin
    d = GradientLayerQ(4, 8, tanh)
    @test isconcretetype(fieldtype(typeof(d), :second_dim))
    @test fieldtype(typeof(d), :second_dim) === Int
    @test d.second_dim === 8

    # The dimension is a type parameter as well as a field, and both are an `Int`: an `Int32`
    # dimension names the same layer as an `Int` one, not a second type that prints alike.
    for (d32, d64) in ((GradientLayerQ(Int32(4), 8, tanh), GradientLayerQ(4, 8, tanh)),
        (GradientLayerP(Int32(4), 8, tanh), GradientLayerP(4, 8, tanh)),
        (ActivationLayerQ(Int32(4), tanh), ActivationLayerQ(4, tanh)),
        (ActivationLayerP(Int32(4), tanh), ActivationLayerP(4, tanh)),
        (LinearLayerQ(Int32(4)), LinearLayerQ(4)),
        (LinearLayerP(Int32(4)), LinearLayerP(4)))
        @test typeof(d32) === typeof(d64)
        @test typeof(d32).parameters[1] isa Int
    end
    @test GradientLayerQ(Int32(4), Int32(8), tanh).second_dim === 8
    @test isconcretetype(fieldtype(typeof(GradientLayerQ(Int32(4), 8, tanh)), :second_dim))

    # `LinearLayerQ` carries the half as its third and last type parameter, so no activation fits
    # after it: a three-parameter alias raises rather than dropping the third.
    @test length(Base.unwrap_unionall(LinearLayerQ).parameters) == 3
    @test_throws "too many parameters" LinearLayerQ{4, 4, typeof(tanh)}
    @test LinearLayerQ{4, 4} === GeometricMachineLearning.LinearLayer{4, 4, :Q}
    @test LinearLayerQ{4, 4} !== LinearLayerP{4, 4}
end
