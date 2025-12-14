module VecMap

export unsupervised

const Constraint = Union{String, Nothing}
const Precision = Union{String, Nothing}
const Seed = Union{Integer, Nothing}
const Threshold = Union{Float64, Nothing}
const CSLS = Union{Integer, Nothing}

function unsupervised(trg_in::String, trg_out::String, src_in::String, src_out::String, constraints::Constraint=nothing, precision::Precision=nothing, encoding::String="", cuda::Bool=false, whiten::Bool=false, batch_size::Integer=-1, seed::Seed=nothing, threshold::Threshold=nothing, csls::CSLS=nothing, verbose::Bool=false)
  flags = []
  if cuda push!(flags, "--cuda") end
  if whiten push!(flags, "--whiten") end
  if verbose push!(flags, "--verbose") end
  if !isnothing(constraints)
    if constraints !== "orthogonal" && constraints !== "unconstrained"
      error("Constraints must be \"orthogonal\" or \"unconstrained\"")
    end
    push!(flags, "--$(constraints)")
  end
  if !isnothing(precision)
    if precision !== "fp16" && precision !== "fp32" && precision !== "fp64"
      error("Precision must be \"fp16\", \"fp32\", or \"fp64\"")
    end
    push!(flags, "--precision $(precision)")
  end
  if !isnothing(threshold) push!(flags, "--threshold $(threshold)") end
  if batch_size > 0 push!(flags, "--batch_size $(batch_size)") end
  if encoding !== "" push!(flags, "--encoding $(encoding)") end
  if !isnothing(seed) push!(flags, "--seed $(seed)") end
  if !isnothing(csls) push!(flags, "--csls $(csls)") end
  run("python3 python/map_embeddings.py --unsupervised $(src_in) $(trg_in) $(src_out) $(trg_out) $(join(flags, " "))"; wait=true)
end

end
