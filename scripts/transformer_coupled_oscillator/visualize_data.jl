using Plots

include("generate_data.jl")
# here the second point mass is altered
params_collection = (   (m1=2, m2=1, k1=1.5, k2=0.3, k=0.5),
                        (m1=2, m2=1, k1=1.5, k2=0.3, k=1.0),
                        (m1=2, m2=1, k1=1.5, k2=0.3, k=1.5),
                        (m1=2, m2=1, k1=1.5, k2=0.3, k=2.0)
                    )

initial_conditions_collection = ( (q=[1.,0.], p=[2.,0.]), )

t_integration_plots = 20

function plot_curves(data_tensor::AbstractArray{T, 3}, one_plot=true) where T 
    q₁ = data_tensor[1,:,:]
    q₂ = data_tensor[2,:,:]
    p₁ = data_tensor[3,:,:]
    p₂ = data_tensor[4,:,:]
    n_param_sets = length(params_collection)
    plot_q₁ = one_plot ? plot(q₁') : plot(q₁', layout=(n_param_sets, 1))
    plot_q₂ = one_plot ? plot(q₂') : plot(q₂', layout=(n_param_sets, 1))
    plot_p₁ = one_plot ? plot(p₁') : plot(p₁', layout=(n_param_sets, 1))
    plot_p₂ = one_plot ? plot(p₂') : plot(p₁', layout=(n_param_sets, 1))
    png(plot_q₁, "q1")
    png(plot_q₂, "q2")
    png(plot_p₁, "p1")
    png(plot_p₂, "p2")
end

data = generate_data(params_collection, initial_conditions_collection, t_integration_plots)
plot_curves(data)