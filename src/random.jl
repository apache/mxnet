function rand!(low::Real, high::Real, out::NDArray)
  # XXX: note we reverse shape because julia and libmx has different dim order
  _sample_uniform(NDArray, low=low, high=high, shape=reverse(size(out)), out=out)
end
function rand{N}(low::Real, high::Real, shape::NTuple{N, Int})
  rand(low, high, shape, cpu())
end
function rand{N}(low::Real, high::Real, shape::NTuple{N, Int}, ctx::Context)
  out = empty(shape, ctx)
  rand!(low, high, out)
end

function randn!(mean::Real, stdvar::Real, out::NDArray)
  # XXX: note we reverse shape because julia and libmx has different dim order
  _sample_normal(NDArray, loc=mean, scale=stdvar, shape=reverse(size(out)), out=out)
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
