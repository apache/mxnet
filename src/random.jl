function rand!(low::Real, high::Real, out::NDArray)
  _random_uniform(low, high, out)
end
function rand(low::Real, high::Real, shape::Tuple, ctx::Context=DEFAULT_CONTEXT)
  out = empty(shape, ctx)
  rand!(low, high, out)
end

function randn!(mean::Real, stdvar::Real, out::NDArray)
  _random_gaussian(mean, stdvar, out)
end
function randn(mean::Real, stdvar::Real, shape::Tuple, ctx::Context=DEFAULT_CONTEXT)
  out = empty(shape, ctx)
  randn!(mean, stdvar, out)
end

function srand!(seed_state::Int)
  @mxcall(:MXRandomSeed, (Cint,), seed_state)
end
