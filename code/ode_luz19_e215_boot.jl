using StatsBase,
DifferentialEquations, 
DiffEqParamEstim, 
ParameterizedFunctions,
ForwardDiff, 
OptimizationOptimJL, 
Optimization,
CSV,
DataFrames,
LineSearches

# Define a timeout macro
# macro timeout(seconds, expr, fail)
#     quote
#         tsk = @task $expr
#         schedule(tsk)
#         Timer($seconds) do timer
#             istaskdone(tsk) || Base.throwto(tsk, InterruptException())
#         end
#         try
#             fetch(tsk)
#         catch _
#             $fail
#         end
#     end
# end

# macro timeout(seconds, expr, fail)
#     quote
#         tsk = @task $esc(expr)
#         schedule(tsk)
#         Timer($(esc(seconds))) do timer
#             istaskdone(tsk) || Base.throwto(tsk, InterruptException())
#         end
#         try
#             fetch(tsk)
#         catch _
#             $(esc(fail))
#         end
#     end
# end

# x = @timeout 1 begin
#     sleep(2)
#     println("done")
#     1
# end "failed"

# Define model
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

luz19_e215 = CSV.read("../ode/luz19_e215_initial.csv", DataFrame; delim = ',')
luz19_e215_init = [collect(Float64, row) for row in eachrow(luz19_e215)]
luz19_e215_N = length(luz19_e215_init)
luz19_e215 = CSV.read("../ode/luz19_e215_fit.csv", DataFrame; delim = ',')
luz19_e215_fit = Array(luz19_e215)
data_times = 0.0:0.1:((size(luz19_e215_fit)[1] - 1) * 0.1)
luz19_e215 = CSV.read("../ode/luz19_e215_fit_w.csv", DataFrame; delim = ',')
luz19_e215_fit_w = Array(luz19_e215)

function bootstrap(B)
    println("start")
    lower = [0, 0]
    upper = [0.01, 1.0]
    # all parameters should be positive, so -1 indicates failed optimization
    # we want 4 parameters: a b s k
    out = -1 * ones(B, 2)
    for b = 1:B
        # sample with replacement
        sample_batch = sample(1:luz19_e215_N, luz19_e215_N)
        init_batch = luz19_e215_init[sample_batch]
        fit_batch = luz19_e215_fit[:, sample_batch]
        fit_w_batch = luz19_e215_fit_w[:, sample_batch]
        function prob_func(prob, i, repeat)
            ODEProblem(prob.f, luz19_e215_init[i], prob.tspan, prob.p, save_idxs = [9]) # get Bt
        end
        ens_prob = EnsembleProblem(prob, prob_func = prob_func)
        losses = [L2Loss(data_times, fit_batch[:, i], data_weight = fit_w_batch[:, i]) for i in 1:luz19_e215_N]
        loss(sim) = sum(losses[i](sim[i]) for i in 1:luz19_e215_N)
        obj = build_loss_objective(ens_prob, Tsit5(), loss, Optimization.AutoForwardDiff(), trajectories = luz19_e215_N, saveat = data_times)
        optprob = OptimizationProblem(
            obj, 
            [0.000005, 0.6],
            lb = lower,
            ub = upper
        )
        result = "failed"
        try
            result = solve(optprob, BFGS(linesearch=LineSearches.BackTracking()), maxtime = 1200.0)
        catch e
        end
        # if result.retcode == Optimization.ReturnCode.Success
        #     out[b, :] = result.u
        # end
        # result = @timeout 600 begin # max 10min fitting time
        #     try
        #         solve(optprob, BFGS())
        #     catch e
        #         "failed"
        #     end
        # end "failed"
        if result != "failed"
            if result.retcode == Optimization.ReturnCode.Success
                out[b, :] = result.u
            end
        end
    end
    return out
end