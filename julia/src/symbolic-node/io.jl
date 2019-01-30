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

"""
    to_json(s::SymbolicNode)

Convert a `SymbolicNode` into a JSON string.
"""
function to_json(s::SymbolicNode)
  ref_json = Ref{char_p}(0)
  @mxcall(:MXSymbolSaveToJSON, (MX_handle, Ref{char_p}), s, ref_json)
  return unsafe_string(ref_json[])
end

"""
    from_json(repr :: AbstractString, ::Type{SymbolicNode})

Load a `SymbolicNode` from a JSON string representation.
"""
function from_json(repr :: AbstractString, ::Type{SymbolicNode})
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolCreateFromJSON, (char_p, Ref{MX_handle}), repr, ref_hdr)
  return SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end

"""
    load(filename :: AbstractString, ::Type{SymbolicNode})

Load a `SymbolicNode` from a JSON file.
"""
function load(filename :: AbstractString, ::Type{SymbolicNode})
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolCreateFromFile, (char_p, Ref{MX_handle}), filename, ref_hdr)
  return SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end

"""
    save(filename :: AbstractString, node :: SymbolicNode)

Save a `SymbolicNode` to a JSON file.
"""
function save(filename :: AbstractString, node :: SymbolicNode)
  @mxcall(:MXSymbolSaveToFile, (MX_handle, char_p), node, filename)
end
