using GeometricProblems.DoublePendulum: hodeproblem
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricMachineLearning

initial_conditions = (
        ([π / 4, π / 2], [0.0, π / 8]), 
        ([π / 4, π / 1], [0.0, 0.0  ]),
        ([π / 1, π / 2], [0.0, π    ])
)

problems = [hodeproblem(initial_condition...) for initial_condition in initial_conditions]
solutions = [integrate(problem, ImplicitMidpoint()) for problem in problems]

sys_dim, input_time_steps, n_params = length(solutions[1].q[0]), length(solutions[1].t), length(initial_conditions)

data = (q = zeros(sys_dim, input_time_steps, n_params), p = zeros(sys_dim, input_time_steps, n_params))

for (solution, i) in zip(solutions, axes(solutions, 1))
    for dim in 1:sys_dim 
        data.q[dim, :, i] = solution.q[:, dim]
        data.p[dim, :, i] = solution.p[:, dim]
    end 
end

dl = DataLoader(data)