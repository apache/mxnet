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
