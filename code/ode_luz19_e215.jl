# using Pkg;
# Pkg.add(["DifferentialEquations", "Optimization", "OptimizationPolyalgorithms", "SciMLSensitivity", "Plots"])
# ! means that the function modifies the argument du in place, no output
using DifferentialEquations, DiffEqParamEstim, ParameterizedFunctions, Plots, CSV, DataFrames, ForwardDiff, OptimizationOptimJL, Optimization, ProgressLogging, OptimizationNLopt, OptimizationBBO, LineSearches
ode_phage = @ode_def begin
    dB = 0.880031 * B * (1 - (B + Ba + Bb + Bab) / k) - 382 * B * (Pa + Pb) - (0.025 + 0.0119 + a) * B
    dBia = 382 * B * Pa + 382 * Bb * Pa - 0.147 * Bia
    dBib = 382 * B * Pb + 382 * Ba * Pb - 0.0506 * Bib
    dBa = 0.025 * B - 382 * Ba * Pb - 0.0119 * Ba + 0.716094 * Ba * (1 - (B + Ba + Bb + Bab) / k)
    dBb = 0.0119 * B - 382 * Bb * Pa - 0.025 * Bb + 0.752185 * Bb * (1 - (B + Ba + Bb + Bab) / k)
    dBab = a * B + 0.0119 * Ba + 0.025 * Bb + 0.618321 * Bab * (1 - (B + Ba + Bb + Bab) / k)
    dPa = 100 * 0.147 * Bia - 382 * (B + Bb) * Pa - 0.09 * Pa
    dPb = 200 * 0.0506 * Bib - 382 * (B + Ba) * Pb - 0.09 * Pb
    dBt = (0.880031 * B * (1 - (B + Ba + Bb + Bab) / k) - 382 * B * (Pa + Pb) - (0.025 + 0.0119 + a) * B) + (382 * B * Pa + 382 * Bb * Pa - 0.147 * Bia) + (382 * B * Pb + 382 * Ba * Pb - 0.0506 * Bib) + (0.025 * B - 382 * Ba * Pb - 0.0119 * Ba + 0.716094 * Ba * (1 - (B + Ba + Bb + Bab) / k)) + (0.0119 * B - 382 * Bb * Pa - 0.025 * Bb + 0.752185 * Bb * (1 - (B + Ba + Bb + Bab) / k)) + (a * B + 0.0119 * Ba + 0.025 * Bb + 0.618321 * Bab * (1 - (B + Ba + Bb + Bab) / k))
end a k
prob = ODEProblem(ode_phage, [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.2], (0.0, 28.0), [0.000005, 0.5])
initial_sol = solve(prob, saveat = 0.1)
# plot(initial_sol)
# read in initial conditions
luz19_e215 = CSV.read("../ode/luz19_e215_initial.csv", DataFrame; delim = ',')
luz19_e215_init = [collect(Float64, row) for row in eachrow(luz19_e215)]
luz19_e215_N = length(luz19_e215_init)
luz19_e215 = CSV.read("../ode/luz19_e215_fit.csv", DataFrame; delim = ',')
luz19_e215_fit = Array(luz19_e215)
data_times = 0.0:0.1:((size(luz19_e215_fit)[1] - 1) * 0.1)
luz19_e215 = CSV.read("../ode/luz19_e215_fit_w.csv", DataFrame; delim = ',')
luz19_e215_fit_w = Array(luz19_e215)
# define ensemble problem
function prob_func(prob, i, repeat)
    ODEProblem(prob.f, luz19_e215_init[i], prob.tspan, prob.p, save_idxs = [9]) # get Bt
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
sim = solve(ens_prob, Tsit5(), trajectories = luz19_e215_N, saveat = data_times)
losses = [L2Loss(data_times, luz19_e215_fit[:, i], data_weight = luz19_e215_fit_w[:, i]) for i in 1:luz19_e215_N] # add data weights significantly slow down the fitting
loss(sim) = sum(losses[i](sim[i]) for i in 1:luz19_e215_N)
# save_idxs will use the fifth variable, Bt to compute loss
obj = build_loss_objective(ens_prob, Tsit5(), loss, Optimization.AutoForwardDiff(), trajectories = luz19_e215_N, saveat = data_times)

# use 95% confint for some parameters
# lower = [0.0012, 15.0, 0.0, 0.5]
# upper = [0.0015, 30.0, 0.1, 1.0]

lower = [0, 0]
upper = [0.01, 1.0]
# a b s k
optprob = OptimizationProblem(
    obj, 
    # [0.01, 100.0, 0.15, 0.7],
    # [0.01, 50.0, 0.15, 0.7], #good 0.1 10 w
    # [0.01, 20.0, 0.2, 0.7], #good 0.1 10 w
    # [0.01, 50.0, 0.15, 0.7],# good 0.05 10 w
    # [0.0013, 20, 0.01, 0.7], # good 0.1, 20 # 0.01, 50, 0.1, 0.7 # 0.001, 20, 0.15, 0.7 # 0.001, 20, 0.01, 0.7
    [0.000005, 0.6],
    # [0.01, 100.0, 0.15, 0.7],
    #[0.01, 50.0, 0.15, 0.7] # good 0.1 20
    #[0.01, 100.0, 0.1, 0.7], 
    lb = lower,
    ub = upper
)
print("start fitting")
result = solve(optprob, BFGS(linesearch=LineSearches.BackTracking()))
print(result)