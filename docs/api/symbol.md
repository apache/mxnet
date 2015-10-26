# MXNet.mx

## Internal

---

<a id="method__group.1" class="lexicon_definition"></a>
#### Group(symbols::MXNet.mx.Symbol...)
Create a symbol that groups symbols together

*source:*
[MXNet/src/symbol.jl:77](https://github.com/dmlc/MXNet.jl/tree/7fa151104fb51d7134da60a5084dfa0d240515f0/src/symbol.jl#L77)

---

<a id="method__variable.1" class="lexicon_definition"></a>
#### Variable(name::Union{AbstractString, Symbol})
Create a symbolic variable with the given name

*source:*
[MXNet/src/symbol.jl:70](https://github.com/dmlc/MXNet.jl/tree/7fa151104fb51d7134da60a5084dfa0d240515f0/src/symbol.jl#L70)

---

<a id="method__from_json.1" class="lexicon_definition"></a>
#### from_json(repr::AbstractString,  ::Type{MXNet.mx.Symbol})
Load Symbol from a JSON string representation.

*source:*
[MXNet/src/symbol.jl:240](https://github.com/dmlc/MXNet.jl/tree/7fa151104fb51d7134da60a5084dfa0d240515f0/src/symbol.jl#L240)

---

<a id="method__get_internals.1" class="lexicon_definition"></a>
#### get_internals(self::MXNet.mx.Symbol)
Get a new grouped symbol whose output contains all the internal outputs of this symbol.

*source:*
[MXNet/src/symbol.jl:63](https://github.com/dmlc/MXNet.jl/tree/7fa151104fb51d7134da60a5084dfa0d240515f0/src/symbol.jl#L63)

---

<a id="method__list_auxiliary_states.1" class="lexicon_definition"></a>
#### list_auxiliary_states(self::MXNet.mx.Symbol)
List all auxiliary states in the symbool.

Auxiliary states are special states of symbols that do not corresponds to an argument,
and do not have gradient. But still be useful for the specific operations.
A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
Most operators do not have Auxiliary states.


*source:*
[MXNet/src/symbol.jl:58](https://github.com/dmlc/MXNet.jl/tree/7fa151104fb51d7134da60a5084dfa0d240515f0/src/symbol.jl#L58)

---

<a id="method__load.1" class="lexicon_definition"></a>
#### load(filename::AbstractString,  ::Type{MXNet.mx.Symbol})
Load Symbol from a JSON file.

*source:*
[MXNet/src/symbol.jl:247](https://github.com/dmlc/MXNet.jl/tree/7fa151104fb51d7134da60a5084dfa0d240515f0/src/symbol.jl#L247)

---

<a id="method__to_json.1" class="lexicon_definition"></a>
#### to_json(self::MXNet.mx.Symbol)
Save Symbol into a JSON string

*source:*
[MXNet/src/symbol.jl:233](https://github.com/dmlc/MXNet.jl/tree/7fa151104fb51d7134da60a5084dfa0d240515f0/src/symbol.jl#L233)

