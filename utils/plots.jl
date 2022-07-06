using GeometricIntegrators
using LaTeXStrings
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


function plot_network_sim(H, H̃, ∇H, ∇H̃, total_loss; xmin=-1.2, xmax=+1.2, ymin=-1.2, ymax=+1.2, nsamples=100, filename=nothing)
    # get offset of learned Hamiltonian
    H̃₀ = H̃([0,0])

    # time step and initial conditions
    Δt = 0.1
    nt = 100
    x₀ = [0.0, 1.0]

    # Hamiltonian vector fields
    v(t,x,v) = v .= ∇H(x)
    ṽ(t,x,v) = v .= ∇H̃(x)

    # compute reference trajectory
    sol_ref = integrate(ODE(v, x₀), TableauGLRK(2), Δt, nt)

    # compute learned trajectory
    sol_hnn = integrate(ODE(ṽ, x₀), TableauGLRK(2), Δt, nt)

    # plot contour lines of Hamiltonian
    X = range(xmin, stop=xmax, length=nsamples)
    Y = range(ymin, stop=ymax, length=nsamples)
    plt_cnt = contour(X, Y, [H([x,y]) for x in X, y in Y], linewidth = 0, fill = true, levels = 7, c = cgrad(:default, rev = true), title=L"$H(q,p)$", xguide=L"$q$", yguide=L"$p$")

    # plot contour lines of learned Hamiltonian
    contour!(plt_cnt, X, Y, [H̃([x,y]) - H̃₀ for x in X, y in Y], linecolor = :black)

    # plot total loss
    plt_loss = plot(total_loss, xguide=L"$n_{\mathrm{training}}$", yguide="Total Loss", yaxis=:log, legend=false)

    # plot solutions
    # plt_sim = contourf(X, Y, [H([x,y]) for x in X, y in Y], linewidth = 0, levels = 7, c = cgrad(:default, rev = true))
    plt_sim = plot(; xguide=L"$q$", yguide=L"$p$")
    plot!(plt_sim, sol_ref.q[1,:], sol_ref.q[2,:], label="Reference")
    plot!(plt_sim, sol_hnn.q[1,:], sol_hnn.q[2,:], label="HNN")

    # Hamiltonians
    H₀ = H([sol_ref.q[1,0], sol_ref.q[2,0]])
    H̃₀ = H̃([sol_hnn.q[1,0], sol_hnn.q[2,0]])
    plt_err = plot(; xguide=L"$t$", yguide=L"$\Delta H(q(t))$", legend=:none)
    plot!(plt_err, sol_ref.t, H.(sol_ref.q) .- H₀, label="Reference")
    plot!(plt_err, sol_hnn.t, H̃.(sol_hnn.q) .- H̃₀, label="HNN")

    l = @layout grid(2, 2, heights=[0.7, 0.3])

    plt = plot(plt_cnt, plt_sim, plt_loss, plt_err, layout = l)

    if filename !== nothing
        savefig(filename)
    end

    return plt
end

