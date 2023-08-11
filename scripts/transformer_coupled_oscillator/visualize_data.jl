using Plots

include("generate_data.jl")
# here the second point mass is altered
params_collection = (   (m1=2, m2=1, k1=1.5, k2=0.3, k=0.0),
                        (m1=2, m2=1, k1=1.5, k2=0.3, k=0.5),
                        (m1=2, m2=1, k1=1.5, k2=0.3, k=0.75),
                        (m1=2, m2=1, k1=1.5, k2=0.3, k=1.0),
                        #(m1=2, m2=1, k1=1.5, k2=0.3, k=1.5),
                        (m1=2, m2=1, k1=1.5, k2=0.3, k=2.0),
                        #(m1=2, m2=1, k1=1.5, k2=0.3, k=2.5),
                        (m1=2, m2=1, k1=1.5, k2=0.3, k=3.0), 
                        #(m1=2, m2=1, k1=1.5, k2=0.3, k=3.5),
                        (m1=2, m2=1, k1=1.5, k2=0.3, k=4.0)
                    )

initial_conditions_collection = ( (q=[1.,0.], p=[2.,0.]), )

t_integration_plots = 100

function plot_curves(data_tensor::AbstractArray{T, 3}, one_plot=true, psize=(1500,1000)) where T 
    q₁ = data_tensor[1,:,:]
    q₂ = data_tensor[2,:,:]
    p₁ = data_tensor[3,:,:]
    p₂ = data_tensor[4,:,:]
    h = t_integration_plots/(size(q₁, 2) - 1)
    t = 0.0:h:t_integration_plots
    n_param_sets = length(params_collection)
    labels = reshape(["k = "*string(params.k) for params in params_collection], 1, n_param_sets)
    plot_q₁ = one_plot ? plot(t, q₁', size=psize) : plot(t, q₁', layout=(n_param_sets, 1), size=psize, label=labels, legend=:topright)
    plot_q₂ = one_plot ? plot(t, q₂', size=psize) : plot(t, q₂', layout=(n_param_sets, 1), size=psize, label=labels, legend=:topright)
    plot_p₁ = one_plot ? plot(t, p₁', size=psize) : plot(t, p₁', layout=(n_param_sets, 1), size=psize, label=labels, legend=:topright)
    plot_p₂ = one_plot ? plot(t, p₂', size=psize) : plot(t, p₁', layout=(n_param_sets, 1), size=psize, label=labels, legend=:topright)
    png(plot_q₁, "q1")
    png(plot_q₂, "q2")
    png(plot_p₁, "p1")
    png(plot_p₂, "p2")
end

data = generate_data(params_collection, initial_conditions_collection, t_integration_plots)
plot_curves(data, false)

