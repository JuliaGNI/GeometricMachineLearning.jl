
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


function rewrite(fun, SV, SX, ti, nn)
    for e in develop(SV)
        str_symbol = replace(string(e), r"\[.*"=>"")
        track = get_track(SV, e, "nt")[2]
        fun = Meta.parse(replace(string(fun), str_symbol => track))
    end
    for e in develop(SX)
        str_symbol = replace(string(e), r"\[.*"=>"")
        track = get_track(SX, e, "sargs")[2]
        fun = Meta.parse(replace(string(fun), str_symbol => track))
    end
    #fun = Meta.parse(replace(string(fun), "SX" => "X"))
    fun = Meta.parse(replace(string(fun), r"function .*" => string("function ∇loss_single(::",typeof(ti),", ::",typeof(nn) ,", sargs, nt)\n")))
end


