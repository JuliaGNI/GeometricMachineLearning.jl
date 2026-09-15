using CairoMakie
using LaTeXStrings
using StatsBase

# Each plot comes in two forms, because a `Figure` cannot be nested inside another one: a `plot_*!`
# that draws into a grid position of an existing figure and returns its axis, and a `plot_*` that
# wraps that in a figure of its own. `plot_result`, which composes four panels, uses the former;
# call the latter to get a single plot on its own.
"Run `f` on the single cell of a new figure of the given size, and return that figure."
function _standalone(f, figsize)
    fig = Figure(size = figsize)
    f(fig[1, 1])
    fig
end

"The full `sym` (`:q` or `:p`) time series of trajectory `i`, as stored by a `DataLoader`."
function _trajectory(dl::DataLoader, sym::Symbol, i::Integer)
    vec(dl.input[sym][:, :, i])
end

function plot_data!(gp, dl::DataLoader,
        title::String = ""; index::AbstractArray = 1:dl.n_params)
    ax = Axis(gp; title = title, titlesize = 15, xlabel = L"q", ylabel = L"p",
        xlabelsize = 14, ylabelsize = 14, limits = ((-3.5, 3.5), (-2.5, 2.5)))

    for i in index
        lines!(ax, _trajectory(dl, :q, i), _trajectory(dl, :p, i);
            label = "Training data "*string(i), linewidth = 3)
    end

    axislegend(ax; position = :lb, nbanks = 2)

    return ax
end

function plot_data(dl::DataLoader, title::String = ""; kwargs...)
    _standalone(gp -> plot_data!(gp, dl, title; kwargs...), (1000, 1000))
end

function plot_verification!(gp, dl::DataLoader,
        nn::NeuralNetwork; index::AbstractArray = [1])
    ax = Axis(gp; title = "Verifications", titlesize = 15, xlabel = L"q", ylabel = L"p",
        xlabelsize = 14, ylabelsize = 14, limits = ((-3.5, 3.5), (-2.5, 2.5)))

    for i in index
        lines!(ax, _trajectory(dl, :q, i), _trajectory(dl, :p, i);
            label = "Training data "*string(i), linewidth = 3)
        init_con = (q = dl.input.q[:, 1, i], p = dl.input.p[:, 1, i])
        prediction = iterate(nn, init_con; n_points = dl.input_time_steps)
        scatterlines!(ax, vec(prediction.q), vec(prediction.p);
            label = "Learned trajectory "*string(i), alpha = 0.8)
    end

    axislegend(ax; position = :lb, nbanks = 2, labelsize = 10)

    return ax
end

function plot_verification(dl::DataLoader, nn::NeuralNetwork; kwargs...)
    _standalone(gp -> plot_verification!(gp, dl, nn; kwargs...), (1000, 800))
end

function plot_loss()
end

function plot_prediction!(gp, dl::DataLoader, nn::NeuralNetwork,
        initial_cond::AbstractArray, H; scale = 1)
    xmin = -3.5*scale
    xmax = 3.5*scale
    ymin = -2.5*scale
    ymax = 2.5*scale

    ax = Axis(gp; title = "Predictions", titlesize = 15, xlabel = L"q", ylabel = L"p",
        xlabelsize = 14, ylabelsize = 14, limits = ((xmin, xmax), (ymin, ymax)))

    X = range(xmin, stop = xmax, length = 100)
    Y = range(ymin, stop = ymax, length = 100)
    contourf!(
        ax, X, Y, [H([x, y]) for x in X, y in Y]; levels = 7, colormap = Reverse(:viridis))

    i = 0
    for qp0 in initial_cond
        i += 1
        valuation = iterate(nn, qp0; n_points = 100)
        scatterlines!(ax, valuation[1, :], valuation[2, :];
            label = "Prediction "*string(i), alpha = 0.8)
    end

    axislegend(ax; position = :lb, nbanks = 2, labelsize = 10)

    return ax
end

function plot_prediction(dl::DataLoader, nn::NeuralNetwork,
        initial_cond::AbstractArray, H; kwargs...)
    _standalone(gp -> plot_prediction!(gp, dl, nn, initial_cond, H; kwargs...), (
        1000, 800))
end

function plot_result(dl::DataLoader, nn::NeuralNetwork, hamiltonian;
        batch_nb_trajectory::Int = dl.n_params,
        batch_verif::Int = 3, filename = nothing, nb_prediction = 2)
    initial_conditions = [(q = dl.input.q[1, 1, i], p = dl.input.p[1, 1, i])
                          for i in 1:dl.n_params]
    min_q = minimum(ic.q for ic in initial_conditions)
    min_p = minimum(ic.p for ic in initial_conditions)
    max_q = maximum(ic.q for ic in initial_conditions)
    max_p = maximum(ic.p for ic in initial_conditions)

    initial_cond = [[
                        linear_trans(rand(), min_q, max_q), linear_trans(rand(), min_p, max_p)]
                    for _ in 1:nb_prediction]
    initial_cond_far = [[linear_trans(rand(), 10*min_q, 10*max_q),
                            linear_trans(rand(), 10*min_p, 10*max_p)]
                        for _ in 1:nb_prediction]

    # the four panels, two by two
    plt = Figure(size = (2000, 1600))

    plot_data!(plt[1, 1],
        dl,
        "Datas";
        index = sort!(sample(1:dl.n_params, batch_nb_trajectory, replace = false)))
    plot_verification!(plt[1, 2], dl, nn;
        index = sort!(sample(1:dl.n_params, batch_verif, replace = false)))
    plot_prediction!(plt[2, 1], dl, nn, initial_cond, hamiltonian)
    plot_prediction!(plt[2, 2], dl, nn, initial_cond_far, hamiltonian; scale = 10)

    if filename !== nothing
        CairoMakie.save(filename, plt)
    end

    return plt
end

linear_trans(x, a, b) = x * (b-a) + a
