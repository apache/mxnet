"""
    rand!(low, high, arr::NDArray)

Draw random samples from a uniform distribution.
Samples are uniformly distributed over the half-open interval [low, high)
(includes low, but excludes high).

#  Examples

```julia
julia> mx.rand(0, 1, mx.zeros(2, 2)) |> copy
2×2 Array{Float32,2}:
 0.405374  0.321043
 0.281153  0.713927
```
"""
function rand!(low::Real, high::Real, out::NDArray)
  _random_uniform(NDArray, low=low, high=high, shape=size(out), out=out)
end

"""
    rand(low, high, shape, context=cpu())

Draw random samples from a uniform distribution.
Samples are uniformly distributed over the half-open interval [low, high)
(includes low, but excludes high).

#  Examples

```julia
julia> mx.rand(0, 1, (2, 2)) |> copy
2×2 Array{Float32,2}:
 0.405374  0.321043
 0.281153  0.713927
```
"""
function rand{N}(low::Real, high::Real, shape::NTuple{N, Int}, ctx::Context=cpu())
  out = empty(shape, ctx)
  rand!(low, high, out)
end

"""
    randn!(mean, std, arr::NDArray)

Draw random samples from a normal (Gaussian) distribution.
"""
function randn!(mean::Real, stdvar::Real, out::NDArray)
  _random_normal(NDArray, loc=mean, scale=stdvar, shape=size(out), out=out)
end

"""
    randn(mean, std, shape, context=cpu())

Draw random samples from a normal (Gaussian) distribution.
"""
function randn{N}(mean::Real, stdvar::Real, shape::NTuple{N,Int}, ctx::Context=cpu())
  out = empty(shape, ctx)
  randn!(mean, stdvar, out)
end

"""
    srand(seed::Int)

Set the random seed of libmxnet
"""
function srand(seed_state::Int)
  @mxcall(:MXRandomSeed, (Cint,), seed_state)
end
