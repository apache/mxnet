# MXNet.mx

## Internal

---

<a id="method__get_internals.1" class="lexicon_definition"></a>
#### get_internals(self::MXNet.mx.Symbol)
Get a new grouped symbol whose output contains all the internal outputs of this symbol.

*source:*
[MXNet/src/symbol.jl:63](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/symbol.jl#L63)

---

<a id="method__group.1" class="lexicon_definition"></a>
#### group(symbols::MXNet.mx.Symbol...)
Create a symbol that groups symbols together

*source:*
[MXNet/src/symbol.jl:77](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/symbol.jl#L77)

---

<a id="method__list_auxiliary_states.1" class="lexicon_definition"></a>
#### list_auxiliary_states(self::MXNet.mx.Symbol)
List all auxiliary states in the symbool.

Auxiliary states are special states of symbols that do not corresponds to an argument,
and do not have gradient. But still be useful for the specific operations.
A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
Most operators do not have Auxiliary states.


*source:*
[MXNet/src/symbol.jl:58](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/symbol.jl#L58)

---

<a id="method__variable.1" class="lexicon_definition"></a>
#### variable(name::Union{AbstractString, Symbol})
Create a symbolic variable with the given name

*source:*
[MXNet/src/symbol.jl:70](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/symbol.jl#L70)

