function rand!(low::Real, high::Real, out::NDArray)
  _random_uniform(low, high, out)
end
function rand{N}(low::Real, high::Real, shape::NTuple{N, Int})
  rand(low, high, shape, cpu())
end
function rand{N}(low::Real, high::Real, shape::NTuple{N, Int}, ctx::Context)
  out = empty(shape, ctx)
  rand!(low, high, out)
end

function randn!(mean::Real, stdvar::Real, out::NDArray)
  _random_gaussian(mean, stdvar, out)
end
function randn{N}(mean::Real, stdvar::Real, shape::NTuple{N,Int})
  randn(mean, stdvar, shape, cpu())
end
function randn{N}(mean::Real, stdvar::Real, shape::NTuple{N,Int}, ctx::Context)
  out = empty(shape, ctx)
  randn!(mean, stdvar, out)
end

function srand!(seed_state::Int)
  @mxcall(:MXRandomSeed, (Cint,), seed_state)
end
