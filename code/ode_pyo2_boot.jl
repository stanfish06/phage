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
    dB = 0.880031 * B * (1 - (B + Br) / k) - b * B * P - a * B
    dBi = b * B * P - s * Bi
    dBr = a * B + 0.730773 * Br * (1 - (B + Br) / k)
    dP = 100 * s * Bi - b * B * P - 0.09 * P
    dBt = 0.880031 * B * (1 - (B + Br) / k) - s * Bi + 0.730773 * Br * (1 - (B + Br) / k)
end a b s k
prob = ODEProblem(ode_phage, [0.2, 0.0, 0.0, 0.2, 0.2], (0.0, 28.0), [0.01, 20.0, 0.2, 0.7])

pyo2 = CSV.read("../ode/pyo2_initial.csv", DataFrame; delim = ',')
pyo2_init = [collect(Float64, row) for row in eachrow(pyo2)]
pyo2_N = length(pyo2_init)
pyo2 = CSV.read("../ode/pyo2_fit.csv", DataFrame; delim = ',')
pyo2_fit = Array(pyo2)
data_times = 0.0:0.1:((size(pyo2_fit)[1] - 1) * 0.1)
pyo2 = CSV.read("../ode/pyo2_fit_w.csv", DataFrame; delim = ',')
pyo2_fit_w = Array(pyo2)

function bootstrap(B)
    println("start")
    lower = [0.0, 100.0, 0.1, 0.5]
    upper = [0.5, 600.0, 1.0, 1.0]
    # all parameters should be positive, so -1 indicates failed optimization
    # we want 4 parameters: a b s k
    out = -1 * ones(B, 4)
    for b = 1:B
        # sample with replacement
        sample_batch = sample(1:pyo2_N, pyo2_N)
        init_batch = pyo2_init[sample_batch]
        fit_batch = pyo2_fit[:, sample_batch]
        fit_w_batch = pyo2_fit_w[:, sample_batch]
        function prob_func(prob, i, repeat)
            ODEProblem(prob.f, init_batch[i], prob.tspan, prob.p, save_idxs = [5]) # get Bt
        end
        ens_prob = EnsembleProblem(prob, prob_func = prob_func)
        losses = [L2Loss(data_times, fit_batch[:, i], data_weight = fit_w_batch[:, i]) for i in 1:pyo2_N]
        loss(sim) = sum(losses[i](sim[i]) for i in 1:pyo2_N)
        obj = build_loss_objective(ens_prob, Tsit5(), loss, Optimization.AutoForwardDiff(), trajectories = pyo2_N, saveat = data_times)
        optprob = OptimizationProblem(
            obj, 
            [0.015, 400.0, 0.2, 0.7],
            lb = lower,
            ub = upper
        )
        result = "failed"
        try
            result = solve(optprob, BFGS(linesearch=LineSearches.BackTracking()))
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