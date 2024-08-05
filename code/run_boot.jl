using Distributed, DataFrames, CSV
n_proc = 20 # 20
addprocs(n_proc)
for m = 2:5
    @everywhere include("ode_luz19_e215_boot.jl")
    b = 10
    samples_pmap = pmap(bootstrap,fill(Int64(b), n_proc))
    out_df = nothing
    for i = 1:n_proc
        df_proc = DataFrame(samples_pmap[i], :auto)
        if i == 1
            out_df = df_proc
        else 
            out_df = vcat(out_df, df_proc)
        end
    end
    # rename!(out_df, [:a, :b, :s, :k])
    rename!(out_df, [:a, :k]) 
    CSV.write("luz19_e215_boot_200_large_b_$(m).csv", out_df)
end