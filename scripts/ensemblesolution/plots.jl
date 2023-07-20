using Plots
using LaTeXStrings

function plots(data::TrainingData, prediction::NamedTuple, batch_nb_trajectory::Int = get_nb_trajectory(data))

    plt = plot([get_data(data,:q, 1,n) for n in 1:get_length_trajectory(data,1)], [get_data(data,:p, 1,n) for n in 1:get_length_trajectory(data,1)], label="Training data.",linewidth = 3,mk=*)

    for i in 2:min(get_nb_trajectory(data), batch_nb_trajectory)
        plot!([get_data(data,:q, i, n) for n in 1:get_length_trajectory(data,i)], [get_data(data,:p, i,n) for n in 1:get_length_trajectory(data,i)], label="Training data.",linewidth = 3,mk=*)
    end


    plot!(plt, prediction[:q], prediction[:p], label="Learned trajectory.", linewidth = 3, guidefontsize=18, tickfontsize=10, size=(1000,800), legendfontsize=15, titlefontsize=15)
    title!("G-SympNet prediction for the simple pendulum")
    xlabel!(L"q")
    ylabel!(L"p")
    plot!(legend=:outerbottom,legendcolumns=2)
    
    
end




