using Plots
using LaTeXStrings
using StatsBase




function plot_data(data::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}}, title::String = ""; index::AbstractArray = 1:get_nb_trajectory(data))

    plt = plot(size=(1000,1000), titlefontsize=15, guidefontsize=14)

    for i in index
        plot!(vcat([get_data(data,:q, i, n) for n in 1:get_length_trajectory(data,i)]...), vcat([get_data(data,:p, i,n) for n in 1:get_length_trajectory(data,i)]...), label="Training data "*string(i),linewidth = 3,mk=*)
    end

    title!(title)

    xlabel!(L"q")
    xlims!((-3.5,3.5))
    ylabel!(L"p")
    ylims!((-2.5,2.5))

    plot!(legend=:outerbottom,legendcolumns=2)

    return plt
     
end


function plot_verification(data::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}}, nns::NeuralNetSolution; index::AbstractArray = [1])

    plt = plot(size=(1000,800), titlefontsize=15, legendfontsize=10, guidefontsize=14)

    for i in index
        plot!(vcat([get_data(data,:q, i, n) for n in 1:get_length_trajectory(data,i)]...), vcat([get_data(data,:p, i,n) for n in 1:get_length_trajectory(data,i)]...), label="Training data "*string(i),linewidth = 3,mk=*)
        q = []
        p = []
        qp = [get_data(data,:q,i,1)..., get_data(data,:p,i,1)...]
        push!(q,qp[1])
        push!(p,qp[2])
        for _ in 2:get_length_trajectory(data,i)
            qp = nns.nn(qp)
            push!(q,qp[1])
            push!(p,qp[2])
        end
        scatter!(q,p, label="Learned trajectory "*string(i), mode="markers+lines", ma = 0.8)
    end

    xlabel!(L"q")
    xlims!((-3.5,3.5))
    ylabel!(L"p")
    ylims!((-2.5,2.5))

    title!("Verifications")

    plot!(legend=:outerbottom,legendcolumns=2)

    return plt
end


function plot_loss()

    

end




function plot_prediction(data::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}}, nns::NeuralNetSolution, initial_cond::AbstractArray, H; scale = 1)

    plt = plot(size=(1000,800), titlefontsize=15, legendfontsize=10, guidefontsize=14)

    xmin = -3.5*scale
    xmax = 3.5*scale
    ymin = -2.5*scale
    ymax = 2.5*scale
    xlabel!(L"q")
    xlims!((xmin,xmax))
    ylabel!(L"p")
    ylims!((ymin,ymax))

    X = range(xmin, stop=xmax, length=100)
    Y = range(ymin, stop=ymax, length=100)
    contour!(X, Y, [H([x,y]) for y in Y, x in X], linewidth = 0, fill = true, levels = 7, c = cgrad(:default, rev = true))



    arrow_indices = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    i=0
    for qp0 in initial_cond
        i+=1
        q = []
        p = []
        qp = qp0
        push!(q,qp[1])
        push!(p,qp[2])
        for _ in 2:100
            qp = nns.nn(qp)
            push!(q,qp[1])
            push!(p,qp[2])
        end
        scatter!(q,p, label="Prediction "*string(i), mode="markers+lines", ma = 0.8)
        #quiver!(q[arrow_indices], p[arrow_indices], quiver=(0.2, 0.2, :auto))
    end



    title!("Predictions")

    plot!(legend=:outerbottom,legendcolumns=2)

    return plt
end

function plot_result(data::TrainingData, nns::NeuralNetSolution, hamiltonian; batch_nb_trajectory::Int = get_nb_trajectory(data), batch_verif::Int = 3, filename = nothing, nb_prediction = 2)

    plt_data = plot_data(data, "Datas"; index = sort!(sample(1:get_nb_trajectory(data), batch_nb_trajectory, replace = false)))

    plt_verif = plot_verification(data, nns; index = sort!(sample(1:get_nb_trajectory(data), batch_verif, replace = false)))

    initial_conditions = [(q = get_data(data,:q,i,1), p = get_data(data,:p,i,1)) for i in 1:get_nb_trajectory(data)]
    min_q = min([initial_conditions[i][:q] for i in 1:get_nb_trajectory(data)]...)
    min_p = min([initial_conditions[i][:p] for i in 1:get_nb_trajectory(data)]...)
    max_q = max([initial_conditions[i][:q] for i in 1:get_nb_trajectory(data)]...)
    max_p = max([initial_conditions[i][:p] for i in 1:get_nb_trajectory(data)]...)

    initial_cond = [[linear_trans(rand(), min_q, max_q)..., linear_trans(rand(), min_p, max_p)...] for _ in 1:nb_prediction]

    plt_pred = plot_prediction(data, nns, initial_cond, hamiltonian)

    initial_cond_far = [[linear_trans(rand(), 10*min_q, 10*max_q)..., linear_trans(rand(), 10*min_p, 10*max_p)...] for _ in 1:nb_prediction]

    plt_farpred = plot_prediction(data, nns, initial_cond_far, hamiltonian; scale = 10)


    l = @layout grid(2, 2)

    plt = plot(plt_data, plt_verif, plt_pred, plt_farpred, layout = l)


    if filename !== nothing
        savefig(filename)
    end

    return plt
end



linear_trans(x,a,b) = x * (b-a) + a
