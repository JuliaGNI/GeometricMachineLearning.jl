
#=
    This function build an expression to compute the gradient of a loss_single functions
=#

function build_gradloss(ti::TrainingIntegrator, nn::AbstractNeuralNetwork, args... ;params = nn.params)

    sargs = symbolic_params(args)[1]           
    sparams = symbolic_params(params)[1] 

    slos = los(ti, nn, sargs..., sparams)
    ∇slos = Symbolics.gradient(slos, rdevelop(sparams))

    sloss = build_function(slos, develop(sargs)..., develop(sparams)... )
    ∇loss = build_function(∇slos, develop(sargs)..., develop(sparams)... )[1]

    return (sloss, ∇loss)

end


