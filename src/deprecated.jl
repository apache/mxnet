# NDArray reshape (#272)
@deprecate reshape(arr::NDArray; shape=()) reshape(arr, shape)
@deprecate Reshape(arr::NDArray; shape=()) reshape(arr, shape)

# SymbolicNode reshape (#279)
@deprecate reshape(sym::SymbolicNode; shape=()) reshape(sym, shape)
@deprecate Reshape(sym::SymbolicNode; shape=()) reshape(sym, shape)

# srand (#282)
@deprecate srand!(seed_state::Int) srand(seed_state)
