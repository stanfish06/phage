# using Pkg;
# Pkg.add(["DifferentialEquations", "Optimization", "OptimizationPolyalgorithms", "SciMLSensitivity", "Plots"])
# ! means that the function modifies the argument du in place, no output
using DifferentialEquations, DiffEqParamEstim, ParameterizedFunctions, Plots, CSV, DataFrames, ForwardDiff, OptimizationOptimJL, Optimization, ProgressLogging, OptimizationNLopt, OptimizationBBO, LineSearches
ode_phage = @ode_def begin
    dB = 0.880031 * B * (1 - (B + Br) / k) - b * B * P - a * B
    dBi = b * B * P - s * Bi
    dBr = a * B + 0.752185 * Br * (1 - (B + Br) / k)
    dP = 200 * s * Bi - b * B * P - 0.09 * P
    dBt = 0.880031 * B * (1 - (B + Br) / k) - s * Bi + 0.752185 * Br * (1 - (B + Br) / k)
end a b s k
prob = ODEProblem(ode_phage, [0.2, 0.0, 0.0, 0.2, 0.2], (0.0, 28.0), [0.0012, 23.0, 0.2, 0.7])
initial_sol = solve(prob, saveat = 0.1)
# plot(initial_sol)
# read in initial conditions
e215 = CSV.read("../ode/e215_initial.csv", DataFrame; delim = ',')
e215_init = [collect(Float64, row) for row in eachrow(e215)]
e215_N = length(e215_init)
e215 = CSV.read("../ode/e215_fit.csv", DataFrame; delim = ',')
e215_fit = Array(e215)
data_times = 0.0:0.1:((size(e215_fit)[1] - 1) * 0.1)
e215 = CSV.read("../ode/e215_fit_w.csv", DataFrame; delim = ',')
e215_fit_w = Array(e215)
# define ensemble problem
function prob_func(prob, i, repeat)
    ODEProblem(prob.f, e215_init[i], prob.tspan, prob.p, save_idxs = [5]) # get Bt
end
# get Bt
# function output_func(sol, i)
#     # sol.u = map(x -> [x[5]], sol.u)
#     sol[1, :] = sol[5, :]
#     sol[2, :] = sol[5, :]
#     sol[3, :] = sol[5, :]
#     sol[4, :] = sol[5, :]
#     sol, false
# end
ens_prob = EnsembleProblem(prob, prob_func = prob_func)
# Check above does what we want, Tsit5 is runge kutta
sim = solve(ens_prob, Tsit5(), trajectories = e215_N, saveat = data_times)
losses = [L2Loss(data_times, e215_fit[:, i], data_weight = e215_fit_w[:, i]) for i in 1:e215_N] # add data weights significantly slow down the fitting
loss(sim) = sum(losses[i](sim[i]) for i in 1:e215_N)
# save_idxs will use the fifth variable, Bt to compute loss
obj = build_loss_objective(ens_prob, Tsit5(), loss, Optimization.AutoForwardDiff(), trajectories = e215_N, saveat = data_times)

# use 95% confint for some parameters
# lower = [0.0012, 15.0, 0.0, 0.5]
# upper = [0.0015, 30.0, 0.1, 1.0]

lower = [0.01, 350.0, 0, 0.5]
upper = [0.015, 400.0, 1.0, 1.0]
# a b s k
optprob = OptimizationProblem(
    obj, 
    # [0.01, 100.0, 0.15, 0.7],
    # [0.01, 50.0, 0.15, 0.7], #good 0.1 10 w
    # [0.01, 20.0, 0.2, 0.7], #good 0.1 10 w
    # [0.01, 50.0, 0.15, 0.7],# good 0.05 10 w
    # [0.0013, 20, 0.01, 0.7], # good 0.1, 20 # 0.01, 50, 0.1, 0.7 # 0.001, 20, 0.15, 0.7 # 0.001, 20, 0.01, 0.7
    [0.012, 370.0, 0.05, 0.7],
    # [0.01, 100.0, 0.15, 0.7],
    #[0.01, 50.0, 0.15, 0.7] # good 0.1 20
    #[0.01, 100.0, 0.1, 0.7], 
    lb = lower,
    ub = upper
)
print("start fitting")
result = solve(optprob, BFGS(linesearch=LineSearches.BackTracking()))
print(result)