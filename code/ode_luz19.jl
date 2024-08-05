# using Pkg;
# Pkg.add(["DifferentialEquations", "Optimization", "OptimizationPolyalgorithms", "SciMLSensitivity", "Plots"])
# ! means that the function modifies the argument du in place, no output
using DifferentialEquations, DiffEqParamEstim, ParameterizedFunctions, Plots, CSV, DataFrames, ForwardDiff, OptimizationOptimJL, Optimization, ProgressLogging, LinearAlgebra, LineSearches
ode_phage = @ode_def begin
    dB = 0.880031 * B * (1 - (B + Br) / k) - b * B * P - a * B
    dBi = b * B * P - s * Bi
    dBr = a * B + 0.716094 * Br * (1 - (B + Br) / k)
    dP = 100 * s * Bi - b * B * P - 0.09 * P
    dBt = 0.880031 * B * (1 - (B + Br) / k) - s * Bi + 0.716094 * Br * (1 - (B + Br) / k)
end a b s k
prob = ODEProblem(ode_phage, [0.2, 0.0, 0.0, 0.2, 0.2], (0.0, 28.0), [0.01, 20.0, 0.15, 0.7])
initial_sol = solve(prob, saveat = 0.1)
# plot(initial_sol)
# read in initial conditions
luz19 = CSV.read("../ode/luz19_initial.csv", DataFrame; delim = ',')
luz19_init = [collect(Float64, row) for row in eachrow(luz19)]
luz19_N = length(luz19_init)
luz19 = CSV.read("../ode/luz19_fit.csv", DataFrame; delim = ',')
luz19_fit = Array(luz19)
data_times = 0.0:0.1:((size(luz19_fit)[1] - 1) * 0.1)
luz19 = CSV.read("../ode/luz19_fit_w.csv", DataFrame; delim = ',')
luz19_fit_w = Array(luz19)
# define ensemble problem
function prob_func(prob, i, repeat)
    ODEProblem(prob.f, luz19_init[i], prob.tspan, prob.p, save_idxs = [5]) # get Bt
end
ens_prob = EnsembleProblem(prob, prob_func = prob_func)
# Check above does what we want, Tsit5 is runge kutta
sim = solve(ens_prob, Tsit5(), trajectories = luz19_N, saveat = data_times)
losses = [L2Loss(data_times, luz19_fit[:, i], data_weight = luz19_fit_w[:, i]) for i in 1:luz19_N] # add data weights significantly slow down the fitting
loss(sim) = sum(losses[i](sim[i]) for i in 1:luz19_N)
# save_idxs will use the fifth variable, Bt to compute loss
obj = build_loss_objective(ens_prob, Tsit5(), loss, Optimization.AutoForwardDiff(), trajectories = luz19_N, saveat = data_times)

# use 95% confint for some parameters
lower = [0.0, 100.0, 0.1, 0.5]
upper = [0.5, 600.0, 1.0, 1.0]
# a b s k
optprob = OptimizationProblem(
    obj, 
    # [0.01, 100.0, 0.15, 0.7],
    # [0.01, 50.0, 0.15, 0.7], #good 0.1 10 w
    # [0.01, 20.0, 0.2, 0.7], #good 0.1 10 w
    # [0.01, 50.0, 0.15, 0.7],# good 0.05 10 w
    [0.015, 400.0, 0.2, 0.7], # good 0.1, 20
    # [0.01, 100.0, 0.15, 0.7],
    #[0.01, 50.0, 0.15, 0.7] # good 0.1 20
    #[0.01, 100.0, 0.1, 0.7], 
    lb = lower,
    ub = upper
)
print("start fitting")
result = solve(optprob, BFGS(linesearch=LineSearches.BackTracking()))


# confidence interval
# first estimate residual se
# define ensemble problem
# Hsse = ForwardDiff.hessian(obj, result.u)
# Hll = inv(Hsse) * 1/2
# luz19 = CSV.read("../ode/luz19_fit_w_homo.csv", DataFrame; delim = ',')
# luz19_fit_w_homo = Array(luz19)
# function prob_func(prob, i, repeat)
#     ODEProblem(prob.f, luz19_init[i], prob.tspan, result.u, save_idxs = [5]) # get Bt
# end
# ens_prob = EnsembleProblem(prob, prob_func = prob_func)
# losses = [L2Loss(data_times, luz19_fit[:, i], data_weight = luz19_fit_w_homo[:, i]) for i in 1:luz19_N]
# sim = solve(ens_prob, Tsit5(), trajectories = luz19_N, saveat = data_times)
# se_est = sqrt(loss(sim) / (sum(luz19_fit_w_homo[:, 1]) - 4))
# se_para = sqrt.(diag(Hll)) * se_est
# 1.96 * se_para