# NDArray reshape (#272)
@deprecate reshape(arr::NDArray; shape=()) reshape(arr, shape)
@deprecate Reshape(arr::NDArray; shape=()) reshape(arr, shape)

# SymbolicNode reshape (#279)
@deprecate reshape(sym::SymbolicNode; shape=()) reshape(sym, shape)
@deprecate Reshape(sym::SymbolicNode; shape=()) reshape(sym, shape)

# srand (#282)
@deprecate srand!(seed_state::Int) srand(seed_state)

# v0.4
@deprecate sin(x::NDArray)    sin.(x)
@deprecate cos(x::NDArray)    cos.(x)
@deprecate tan(x::NDArray)    tan.(x)
@deprecate arcsin(x::NDArray) asin.(x)
@deprecate arccos(x::NDArray) acos.(x)
@deprecate arctan(x::NDArray) atan.(x)

@deprecate sinh(x::NDArray)    sinh.(x)
@deprecate cosh(x::NDArray)    cosh.(x)
@deprecate tanh(x::NDArray)    tanh.(x)
@deprecate arcsinh(x::NDArray) asinh.(x)
@deprecate arccosh(x::NDArray) acosh.(x)
@deprecate arctanh(x::NDArray) atanh.(x)

# @deprecate make `randn` exported accidentially
# so we make the depwarn manually
function randn(μ, σ, dims::NTuple{N,Int}, ctx::Context = cpu()) where N
  warn("mx.randn(μ, σ, dims, ctx = cpu()) is deprecated, use " *
       "mx.randn(dims...; μ = μ, σ = σ, context = ctx) instead.")
  mx.randn(dims...; μ = μ, σ = σ, context = ctx)
end

function randn!(μ, σ, x::NDArray)
  warn("mx.randn!(μ, σ, x::NDArray) is deprecated, use " *
       "mx.randn!(x; μ = μ, σ = σ) instead.")
  randn!(x; μ = μ, σ = σ)
end

function rand!(low::Real, high::Real, x::NDArray)
  warn("rand!(low, high, x::NDArray) is deprecated, use " *
       "rand!(x, low = low, high = high) instead.")
  rand!(x, low = low, high = high)
end

function rand(low::Real, high::Real, dims::NTuple{N,Int}, context::Context = cpu()) where N
  warn("rand!(low, high, dims, x::NDArray, context = cpu()) is deprecated, use " *
       "rand!(dims..., x; low = low, high = high, context = cpu()) instead.")
  rand(dims...; low = low, high = high, context = context)
end
