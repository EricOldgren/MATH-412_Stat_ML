#!/usr/bin/env julia --threads=3
# Run with `julia --threads=3 SVD.jl`
# Needs 5-6GB of RAM

# Params
const N = 100
const ONLY_RGB = true # Set to true to ignore grayscale images.
const ONLY_TRAIN = true # Set to true to ignore validation images.

# Imports
# import Pkg; ["DataFrames", "CSV", "Images", "ProgressMeter", "TSVD", "NPZ"] .|> Pkg.add # Uncomment to install packages
println("Importing libraries...")
using DataFrames, CSV, Images, ProgressMeter, TSVD, NPZ

# Import image metadata
println("Importing image metadata...")
df_images = CSV.read("./data/imagenette2-320/noisy_imagenette.csv", DataFrame);

# Define helper functions
# N0f8 is a Float8 with exponent 0, i.e. it takes values 0, 1/255, ..., 254/255, 1
function image_to_tensor(i::Matrix{RGB{N0f8}})::Array{N0f8, 3}
    local img = @view i[1:320, 1:320]
    return cat(img .|> p -> p.r, img .|> p -> p.g, img .|> p -> p.b, dims=3)
end

function image_to_tensor(i::Matrix{Gray{N0f8}})::Array{N0f8, 3}
    local img = @view i[1:320, 1:320]
    local gs = img .|> p -> p.val
    return cat(gs, gs, gs, dims=3)
end

load_image(idx::Int)::Matrix{RGB{N0f8}} = Images.load("data/imagenette2-320/$(df_images[idx, "path"])")
load_image_as_tensor(idx::Int)::Array{N0f8, 3} = image_to_tensor(load_image(idx))
image_is_rgb(idx::Int)::Bool = typeof(load_image(idx)[1, 1]) <: RGB
image_is_train(idx::Int)::Bool = !df_images[idx, "is_valid"]

# Remove grayscale images if ONLY_RGB is set.
# We do this beforehand, to prevent resizing the "large" data tensor.
# Takes 10-20s
if ONLY_RGB || ONLY_TRAIN
    println("Filtering images...")
    local mask::BitVector = 1:size(df_images, 1) .|> idx -> (!ONLY_TRAIN || image_is_train(idx)) && (!ONLY_RGB || image_is_rgb(idx));
    global df_images = df_images[mask, :]
    GC.gc() # Run garbage collection
    println("Remaining image count: $(size(df_images, 1))")
end

# Pre-allocate output
println("Allocating tensor...")
data::Array{N0f8, 4} = zeros(N0f8, size(df_images, 1), 320, 320, 3);
GC.gc()

# Takes ~1min for the training set
println("Importing images...")
@showprogress for i in 1:size(df_images, 1)
    data[i, :, :, :] = load_image_as_tensor(i)
end
GC.gc()

# Create a reshaped view of the data (does not copy anything)
flat_data::Array{N0f8, 3} = reshape(data, size(data, 1), :, 3);

# Allocate SVD result containers (to be able to run the following loop in parallel for each channel)
Us::Vector{Matrix{Float32}} = Vector(undef, 3)
Σs::Vector{Vector{Float32}} = Vector(undef, 3)
Vs::Vector{Matrix{Float32}} = Vector(undef, 3)

# Run SVD for each channel. May take ~3min for N=10 on 3 threads
# Julia needs to be run with --threads=3 for this to be "fast".
println("Running truncated SVD on each channel with $N components (on $(min(3, Threads.nthreads())) threads)...")
Threads.@threads for i in collect(1:3)
    Us[i], Σs[i], Vs[i] = tsvd((@view flat_data[:, :, i]), N)
end
GC.gc()

U_tensor = cat(Us..., dims=3)
Σ_matrix = hcat(Σs...)
V_tensor = cat((Vs .|> V -> reshape(V, 320, 320, :))..., dims=4)

println("Exporting result...")
if !isdir("svd/")
    mkpath("svd/")
    println("Created svd/ directory.")
end
npzwrite("svd/SVD-$(N)$(ONLY_RGB ? "-rgb" : "-all")$(ONLY_TRAIN ? "-train" : "-all").npz", Dict(
    # "U" => U_tensor, # The U tensor is not used in PCA. Uncomment to save anyway
    "S" => Σ_matrix,
    "V" => V_tensor
))
println("Done.")