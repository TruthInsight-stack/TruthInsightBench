## --- Import datasets
using NetCDF, StatGeochem
cd(@__DIR__)

# Read glacial and nonglacial data files
earth = importdataset("../glacial_erosion_Earth.tsv", '\t', importas=:Tuple);
ngearth = importdataset("../nonglacial_erosion_Earth.tsv", '\t', importas=:Tuple)

# Combine CHELSEA datasets
chelsea = zeros(43200, 20880)
for i in 1:9
    chelsea .+= ncread("CHELSA_pr_0$(i)_1981-2010_V.2.1.nc", "Band1")
end
for i in 10:12
    chelsea .+= ncread("CHELSA_pr_$(i)_1981-2010_V.2.1.nc", "Band1")
end
chelsea./=10

function find_precip(lats, lons)
    @assert eachindex(lats) == eachindex(lons)
    result = fill(NaN, size(lats))
    @inbounds for n in eachindex(lats)
        if lons[n]==lons[n] && lats[n]==lats[n]
            i = floor(Int, (lons[n]+180)*120)+1
            j = floor(Int, (lats[n]+90)*120)+1
            if (0 < i < 43200) && (0 < j < 20880)
                result[n] = chelsea[i,j]
            end
        end
    end
    return result
end

## --- Find precipitation for glacial locations

precipg = find_precip(earth.Latitude, earth.Longitude)

ds = (;Latitude=earth.Latitude,
    Longitude=earth.Longitude,
    Precipitation_mm_yr = precipg,
)

exportdataset(ds, "precipitation_glacial.tsv", '\t')

## --- Find precipitation for nonglacial locations

precipng = find_precip(ngearth.Latitude, ngearth.Longitude)

ds = (;Latitude=ngearth.Latitude,
    Longitude=ngearth.Longitude,
    Precipitation_mm_yr = precipng,
)

exportdataset(ds, "precipitation_nonglacial.tsv", '\t')

## --- End of File
