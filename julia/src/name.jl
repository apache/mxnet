# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
