using CairoMakie
using LaTeXStrings
using StatsBase

# Under Plots each `plot_*` built its own plot object and `plot_result` composed the four into a
# `grid(2, 2)`. Makie cannot nest one `Figure` inside another, so each of them is split in two: a
# `plot_*!` that draws into a grid position and returns its axis, and a `plot_*` that wraps it in a
# figure of its own. `plot_result` uses the former, everything else the latter.
"Run `f` on the single cell of a new figure of the given size, and return that figure."
function _standalone(f, size)
    fig = Figure(size = size)
    f(fig[1, 1])
    fig
end

_trajectory(data, sym, i) = vcat([get_data(data, sym, i, n) for n in 1:get_length_trajectory(data, i)]...)


function plot_data!(gp, data::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}}, title::String = ""; index::AbstractArray = 1:get_nb_trajectory(data))

    ax = Axis(gp; title = title, titlesize = 15, xlabel = L"q", ylabel = L"p",
              xlabelsize = 14, ylabelsize = 14, limits = ((-3.5, 3.5), (-2.5, 2.5)))

    for i in index
        lines!(ax, _trajectory(data, :q, i), _trajectory(data, :p, i);
               label = "Training data "*string(i), linewidth = 3)
    end

    axislegend(ax; position = :lb, nbanks = 2)

    return ax
end

plot_data(data::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}}, title::String = ""; kwargs...) =
    _standalone(gp -> plot_data!(gp, data, title; kwargs...), (1000, 1000))


function plot_verification!(gp, data::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}}, nns::NeuralNetSolution; index::AbstractArray = [1])

    ax = Axis(gp; title = "Verifications", titlesize = 15, xlabel = L"q", ylabel = L"p",
              xlabelsize = 14, ylabelsize = 14, limits = ((-3.5, 3.5), (-2.5, 2.5)))

    for i in index
        lines!(ax, _trajectory(data, :q, i), _trajectory(data, :p, i);
               label = "Training data "*string(i), linewidth = 3)
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
        scatterlines!(ax, q, p; label = "Learned trajectory "*string(i), alpha = 0.8)
    end

    axislegend(ax; position = :lb, nbanks = 2, labelsize = 10)

    return ax
end

plot_verification(data::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}}, nns::NeuralNetSolution; kwargs...) =
    _standalone(gp -> plot_verification!(gp, data, nns; kwargs...), (1000, 800))


function plot_loss()



end


function plot_prediction!(gp, data::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}}, nns::NeuralNetSolution, initial_cond::AbstractArray, H; scale = 1)

    xmin = -3.5*scale
    xmax = 3.5*scale
    ymin = -2.5*scale
    ymax = 2.5*scale

    ax = Axis(gp; title = "Predictions", titlesize = 15, xlabel = L"q", ylabel = L"p",
              xlabelsize = 14, ylabelsize = 14, limits = ((xmin, xmax), (ymin, ymax)))

    X = range(xmin, stop=xmax, length=100)
    Y = range(ymin, stop=ymax, length=100)
    contourf!(ax, X, Y, [H([x,y]) for x in X, y in Y]; levels = 7, colormap = Reverse(:viridis))

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
        scatterlines!(ax, q, p; label = "Prediction "*string(i), alpha = 0.8)
    end

    axislegend(ax; position = :lb, nbanks = 2, labelsize = 10)

    return ax
end

plot_prediction(data::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}}, nns::NeuralNetSolution, initial_cond::AbstractArray, H; kwargs...) =
    _standalone(gp -> plot_prediction!(gp, data, nns, initial_cond, H; kwargs...), (1000, 800))


function plot_result(data::TrainingData, nns::NeuralNetSolution, hamiltonian; batch_nb_trajectory::Int = get_nb_trajectory(data), batch_verif::Int = 3, filename = nothing, nb_prediction = 2)

    initial_conditions = [(q = get_data(data,:q,i,1), p = get_data(data,:p,i,1)) for i in 1:get_nb_trajectory(data)]
    min_q = min([initial_conditions[i][:q] for i in 1:get_nb_trajectory(data)]...)
    min_p = min([initial_conditions[i][:p] for i in 1:get_nb_trajectory(data)]...)
    max_q = max([initial_conditions[i][:q] for i in 1:get_nb_trajectory(data)]...)
    max_p = max([initial_conditions[i][:p] for i in 1:get_nb_trajectory(data)]...)

    initial_cond = [[linear_trans(rand(), min_q, max_q)..., linear_trans(rand(), min_p, max_p)...] for _ in 1:nb_prediction]
    initial_cond_far = [[linear_trans(rand(), 10*min_q, 10*max_q)..., linear_trans(rand(), 10*min_p, 10*max_p)...] for _ in 1:nb_prediction]

    # the `grid(2, 2)` of the Plots version
    plt = Figure(size = (2000, 1600))

    plot_data!(plt[1, 1], data, "Datas"; index = sort!(sample(1:get_nb_trajectory(data), batch_nb_trajectory, replace = false)))
    plot_verification!(plt[1, 2], data, nns; index = sort!(sample(1:get_nb_trajectory(data), batch_verif, replace = false)))
    plot_prediction!(plt[2, 1], data, nns, initial_cond, hamiltonian)
    plot_prediction!(plt[2, 2], data, nns, initial_cond_far, hamiltonian; scale = 10)

    if filename !== nothing
        CairoMakie.save(filename, plt)
    end

    return plt
end



linear_trans(x,a,b) = x * (b-a) + a
