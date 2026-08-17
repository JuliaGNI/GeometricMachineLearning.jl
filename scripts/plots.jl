using CairoMakie
using LaTeXStrings

# Fill colours for the contour panels: the Hamiltonian panels use a reversed viridis and the error
# panel the unreversed one, so that the two read differently at a glance.
const _reversed_colormap = Reverse(:viridis)
const _default_colormap = :viridis

"""
    _contour_panel!(position, X, Y, Z; kwargs...)

A filled contour of `Z` over `X × Y`, on its own axis at `position` in the enclosing figure layout.
Returns the axis so a caller can draw a second, unfilled contour on top of it.
"""
function _contour_panel!(position, X, Y, Z; levels = 7, colormap = _reversed_colormap, kwargs...)
    ax = Axis(position; kwargs...)
    contourf!(ax, X, Y, Z; levels = levels, colormap = colormap)
    ax
end

function plot_hnn(H, H̃, total_loss; xmin=-1.2, xmax=+1.2, ymin=-1.2, ymax=+1.2, nsamples=100, filename=nothing)
    #get offset of learned Hamiltonian
    H̃₀ = H̃([0,0])

    X = range(xmin, stop=xmax, length=nsamples)
    Y = range(ymin, stop=ymax, length=nsamples)

    fig = Figure(size = (1000, 800))

    # contour lines of the Hamiltonian, with those of the learned one on top
    ax_cnt = _contour_panel!(fig[1, 1], X, Y, [H([x,y]) for x in X, y in Y];
                             xlabel = L"q", ylabel = L"p")
    contour!(ax_cnt, X, Y, [H̃([x,y]) - H̃₀ for x in X, y in Y]; color = :black)

    # contours of the error of the Hamiltonian, in per cent of its maximum
    m = maximum(H([x,y]) for x in X, y in Y)
    _contour_panel!(fig[1, 2], X, Y, 100*[(H̃([x,y]) - H̃₀ - H([x,y]))/m for x in X, y in Y];
                    colormap = _default_colormap, xlabel = L"q", ylabel = L"p")

    # total loss, across the full width below the two contour panels
    ax_loss = Axis(fig[2, 1:2]; xlabel = "n(training)", ylabel = "Total Loss")
    lines!(ax_loss, total_loss)

    # the loss occupies the bottom 40% of the figure
    rowsize!(fig.layout, 2, Relative(0.4))

    if filename !== nothing
        CairoMakie.save(filename, fig)
    end

    return fig
end


function plot_network_sim(H, H̃, ∇H, ∇H̃, total_loss; xmin=-1.2, xmax=+1.2, ymin=-1.2, ymax=+1.2, nsamples=100, filename=nothing)
    # get offset of learned Hamiltonian
    H̃₀ = H̃([0,0])

    # time step and initial conditions
    Δt = 0.1
    nt = 100
    x₀ = [0.0, 1.0]

    # Hamiltonian vector fields
    v(t,x,v) = v .= ∇H(x)
    ṽ(t,x,v) = v .= ∇H̃(x)

    # compute reference trajectory
    sol_ref = integrate(ODE(v, x₀), TableauGLRK(2), Δt, nt)

    # compute learned trajectory
    sol_hnn = integrate(ODE(ṽ, x₀), TableauGLRK(2), Δt, nt)

    X = range(xmin, stop=xmax, length=nsamples)
    Y = range(ymin, stop=ymax, length=nsamples)

    fig = Figure(size = (1000, 800))

    # contour lines of the Hamiltonian, with those of the learned one on top
    ax_cnt = _contour_panel!(fig[1, 1], X, Y, [H([x,y]) for x in X, y in Y];
                             title = L"$H(q,p)$", xlabel = L"$q$", ylabel = L"$p$")
    contour!(ax_cnt, X, Y, [H̃([x,y]) - H̃₀ for x in X, y in Y]; color = :black)

    # solutions
    ax_sim = Axis(fig[1, 2]; xlabel = L"$q$", ylabel = L"$p$")
    lines!(ax_sim, sol_ref.q[1,:], sol_ref.q[2,:]; label = "Reference")
    lines!(ax_sim, sol_hnn.q[1,:], sol_hnn.q[2,:]; label = "HNN")
    axislegend(ax_sim)

    # total loss
    ax_loss = Axis(fig[2, 1]; xlabel = L"$n_{\mathrm{training}}$", ylabel = "Total Loss",
                   yscale = log10)
    lines!(ax_loss, total_loss)

    # Hamiltonians
    H₀ = H([sol_ref.q[1,0], sol_ref.q[2,0]])
    H̃₀ = H̃([sol_hnn.q[1,0], sol_hnn.q[2,0]])
    ax_err = Axis(fig[2, 2]; xlabel = L"$t$", ylabel = L"$\Delta H(q(t))$")
    lines!(ax_err, sol_ref.t, H.(sol_ref.q) .- H₀)
    lines!(ax_err, sol_hnn.t, H̃.(sol_hnn.q) .- H̃₀)

    # the two contour panels occupy the top 70% of the figure
    rowsize!(fig.layout, 1, Relative(0.7))

    if filename !== nothing
        CairoMakie.save(filename, fig)
    end

    return fig
end
