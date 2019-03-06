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

@inline _wait_to_read(arr :: NDArray) =
  @mxcall(:MXNDArrayWaitToRead, (MX_handle,), arr)
@inline _wait_to_write(arr :: NDArray) =
  @mxcall(:MXNDArrayWaitToWrite, (MX_handle,), arr)

"""
    try_get_shared(arr; sync=:nop)

Try to create a Julia array by sharing the data with the underlying `NDArray`.

# Arguments:

* `arr::NDArray`: the array to be shared.

!!! note
    The returned array does not guarantee to share data with the underlying `NDArray`.
    In particular, data sharing is possible only when the `NDArray` lives on CPU.

* `sync::Symbol`: `:nop`,`:write`, `:read`
  On CPU, invoke `_wait_to_read` if `:read`;
  invoke `_wait_to_write` if `:write`.
"""
function try_get_shared(x::NDArray; sync::Symbol=:nop)
  if context(x).device_type == CPU
    # try to do data sharing
    if sync == :read
      _wait_to_read(x)
    elseif sync == :write
      _wait_to_write(x)
    end

    unsafe_wrap(Array, pointer(x), size(x))
  else
    # impossible to share, just copying
    copy(x)
  end
end

"""
    is_shared(j_arr, arr)

Test whether `j_arr` is sharing data with `arr`.

# Arguments:

* `j_arr::Array`: the Julia Array.
* `arr::NDArray`: the `NDArray`.
"""
is_shared(::Array, ::NDArray) = false

function is_shared(j_arr::Array{T}, arr::NDArray{T}) where {T<:DType}
  if length(j_arr) != length(arr)
    return false
  end
  if context(arr).device_type != CPU
    return false
  end
  pointer(j_arr) == pointer(arr)
end

"""
    load(filename, ::Type{NDArray})

Load NDArrays from binary file.

# Arguments:
* `filename::String`: the path of the file to load. It could be S3 or HDFS address.

Returns either `Dict{Symbol, NDArray}` or `Vector{NDArray}`.

`filename` can point to `s3` or `hdfs` resources if the `libmxnet` is built with the
corresponding components enabled. Examples:
* `s3://my-bucket/path/my-s3-ndarray`
* `hdfs://my-bucket/path/my-hdfs-ndarray`
* `/path-to/my-local-ndarray`
"""
function load(filename::AbstractString, ::Type{<:NDArray})
  out_size      = Ref{MX_uint}(0)
  out_hdrs      = Ref{Ptr{MX_handle}}(0)
  out_name_size = Ref{MX_uint}(0)
  out_names     = Ref{char_pp}(0)
  @mxcall(:MXNDArrayLoad, (char_p, Ref{MX_uint}, Ref{Ptr{MX_handle}}, Ref{MX_uint}, Ref{char_pp}),
          filename, out_size, out_hdrs, out_name_size, out_names)
  out_name_size = out_name_size[]
  out_size      = out_size[]
  if out_name_size == 0
    return [NDArray(MX_NDArrayHandle(hdr)) for hdr in unsafe_wrap(Array, out_hdrs[], out_size)]
  else
    @assert out_size == out_name_size
    return Dict([(Symbol(unsafe_string(k)), NDArray(MX_NDArrayHandle(hdr))) for (k,hdr) in
                 zip(unsafe_wrap(Array, out_names[], out_size), unsafe_wrap(Array, out_hdrs[], out_size))])
  end
end

"""
    save(filename::AbstractString, data)

Save NDarrays to binary file. Filename could be S3 or HDFS address, if `libmxnet` is built
with corresponding support (see `load`).

* `filename::String`: path to the binary file to write to.
* `data`: data to save to file. Data can be a`NDArray`, a `Vector` of `NDArray`,
  or a `Dict{Symbol}` contains `NDArray`s.
"""
save(filename::String, data::NDArray) = save(filename, [data])

save(filename::String, data::VecOfNDArray) =
  @mxcall(:MXNDArraySave, (char_p, MX_uint, Ptr{MX_handle}, char_pp),
          filename, length(data), MX_handle[data...], char_pp(0))

function save(filename::String, data::Dict{Symbol})
  names  = keys(data)
  arrays = MX_handle.(collect(values(data)))
  names  = String.(collect(names))

  @mxcall(:MXNDArraySave, (char_p, MX_uint, Ptr{MX_handle}, char_pp),
          filename, length(names), arrays, names)
end
