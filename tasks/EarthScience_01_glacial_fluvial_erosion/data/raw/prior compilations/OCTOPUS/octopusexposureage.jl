# Calculate basin-averaged exposure ages from OCTOPUS data using CAIRN scalings
# Ref. OCTOPUS: 10.5194/essd-10-2123-2018
# and CAIRN: 10.5194/esurf-4-655-2016
using StatGeochem, Plots
cd(@__DIR__)

# Import OCTOPUS dataset
ds = importdataset("octopusdata.tsv", '\t', importas=:Tuple)

## --- Beryllium-10 ------------------------------------------------------------

# Sea-level high-latitude reference production [at/g/yr]
Be10P_slhl = 4.30
# Al-26 Decay constant [1/yr]
λBe = log(2)/1.39e6

# Total corrected production, in atoms per gram per year
Be10P = @. Be10P_slhl * ds.beprod * ds.betopo * ds.beself * ds.besnow

# Calculate exposure age [yr]
iterations = 2 # Best results for n=1 or 2 additional iterations
Be10exposure = ds.be10nc ./ Be10P
decay = exp.(-λBe.*Be10exposure)
be10ncᵢ = ds.be10nc ./ decay
for _ in 1:iterations
    @. Be10exposure = be10ncᵢ / Be10P
    @. decay = exp(-λBe*Be10exposure)
    @. be10ncᵢ = ds.be10nc / decay
end

# Plot results, to check
h =  plot(Be10exposure, ds.ebe_mmkyr/1000,
    seriestype=:scatter,
    framestyle=:box,
    xscale=:log10,
    yscale=:log10,
    xlabel="Timescale [yr]",
    ylabel="Erosion rate [mm/yr]",
    label="Data",
)
x = collect(xlims())
plot!(h, x, 600.0./x, label="600 mm")
display(h)


## --- Aluminum-26 -------------------------------------------------------------

# Sea-level high-latitude reference production [at/g/yr]
Al26P_slhl = 31.10
# Al-26 Decay constant [1/yr]
λAl = log(2)/7.17e5

# Total corrected production, in atoms per gram per year
Al26P = @. Al26P_slhl * ds.alprod * ds.altopo * ds.alself * ds.alsnow

# Calculate exposure age [yr]
iterations = 2 # Alst results for n=1 or 2 additional iterations
Al26exposure = ds.al26nc ./ Al26P
decay = exp.(-λAl.*Al26exposure)
al26ncᵢ = ds.al26nc ./ decay
for _ in 1:iterations
    @. Al26exposure = al26ncᵢ / Al26P
    @. decay = exp(-λAl*Al26exposure)
    @. al26ncᵢ = ds.al26nc / decay
end

# Plot results, to check
h =  plot(Al26exposure, ds.eal_mmkyr/1000,
    seriestype=:scatter,
    framestyle=:box,
    xscale=:log10,
    yscale=:log10,
    xlabel="Timescale [yr]",
    ylabel="Erosion rate [mm/yr]",
    label="Data",
)
x = collect(xlims())
plot!(h, x, 600.0./x, label="600 mm")
display(h)


## --- Save results

exportdataset((;Be10_exposure_yr=Be10exposure, Al26_exposure_yr=Al26exposure), "octopusages.tsv", '\t')


## --- End of File
