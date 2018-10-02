abstract type AbstractNameManager end
const NameType = Union{Base.Symbol, AbstractString}
const NameCounter = Dict{Base.Symbol, Int}

import Base: get!

# Default implementation for generating a name for a symbol.
# When a name is specified by the user, it will be used. Otherwise, a name
# is automatically generated based on the hint string.
function _default_get_name!(counter :: NameCounter, name :: NameType, hint :: NameType)
  if isa(name, Base.Symbol) || !isempty(name)
    return Symbol(name)
  end

  hint = Symbol(hint)
  if !haskey(counter, hint)
    counter[hint] = 0
  end
  name = Symbol("$hint$(counter[hint])")
  counter[hint] += 1
  return name
end

mutable struct BasicNameManager <: AbstractNameManager
  counter :: NameCounter
end
BasicNameManager() = BasicNameManager(NameCounter())

function get!(manager :: BasicNameManager, name :: NameType, hint :: NameType)
  _default_get_name!(manager.counter, name, hint)
end

mutable struct PrefixNameManager <: AbstractNameManager
  prefix  :: Base.Symbol
  counter :: NameCounter
end
PrefixNameManager(prefix :: NameType) = PrefixNameManager(Symbol(prefix), NameCounter())

function get!(manager :: PrefixNameManager, name :: NameType, hint :: NameType)
  name = _default_get_name!(manager.counter, name, hint)
  return Symbol("$(manager.prefix)$name")
end

DEFAULT_NAME_MANAGER = BasicNameManager()
