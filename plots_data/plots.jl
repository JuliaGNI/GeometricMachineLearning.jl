using Plots

function plot_network(H, H̃, total_loss; xmin=-1.2, xmax=+1.2, ymin=-1.2, ymax=+1.2, nsamples=100, filename=nothing)
    #get offset of learned Hamiltonian
    H̃₀ = H̃([0,0])

    #plot contour lines of Hamiltonian
    X = range(xmin, stop=xmax, length=nsamples)
    Y = range(ymin, stop=ymax, length=nsamples)
    plt_cnt = contour(X, Y, [H([x,y]) for x in X, y in Y], linewidth = 0, fill = true, levels = 7, c = cgrad(:default, rev = true))

    #plot contour lines of learned Hamiltonian
    contour!(plt_cnt, X, Y, [H̃([x,y]) - H̃₀ for x in X, y in Y], linecolor = :black)

    #plot contours of error of Hamiltonian
    plt_err = contourf(X, Y, [H̃([x,y]) - H̃₀ - H([x,y]) for x in X, y in Y])

    # plot total loss
    plt_loss = plot(total_loss, xguide="n(training)", yguide="Total Loss", legend=false, size=(1000,800))

    l = @layout [
            grid(1,2)
            b{0.4h}
    ]

    plt = plot(plt_cnt, plt_err, plt_loss, layout = l)

    if filename !== nothing
        savefig(filename)
    end

    return plt
end
