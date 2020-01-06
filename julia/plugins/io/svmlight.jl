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

#=doc
SVMLight / LibSVM is a popular data format for sparse features. Some preprocessed
datasets in this format could be found at http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
=#
using MXNet
using SVMLightLoader

mutable struct SVMLightProvider <: mx.AbstractDataProvider
  filename   :: AbstractString
  batch_size :: Int
  fea_dim    :: Int
  data_name  :: Symbol
  label_name :: Symbol
end

function SVMLightProvider(filename::AbstractString, batch_size::Int; fea_dim::Int=-1,
                          data_name::Symbol=:data, label_name::Symbol=:label)
  if fea_dim == -1
    info("SVMLightProvider: going over file to get feature dimension of $filename")
    f = SVMLightFile(filename)
    for (data, label) in f
      fea_dim = max(fea_dim, length(data))
    end
  end

  return SVMLightProvider(filename, batch_size, fea_dim, data_name, label_name)
end

mx.get_batch_size(provider :: SVMLightProvider) = provider.batch_size
function mx.provide_data(provider :: SVMLightProvider)
  [(provider.data_name, (provider.fea_dim, provider.batch_size))]
end
function mx.provide_label(provider :: SVMLightProvider)
  [(provider.label_name, (provider.batch_size,))]
end

function mx.eachbatch(provider :: SVMLightProvider)
  data_jl  = zeros(mx.MX_float, (provider.fea_dim, provider.batch_size))
  data_nd  = mx.empty(size(data_jl))
  label_jl = zeros(mx.MX_float, (provider.batch_size,))
  label_nd = mx.empty(size(label_jl))

  batch = mx.DataBatch([data_nd], [label_nd], provider.batch_size)
  function _svmlight_iter()
    f = SVMLightFile(provider.filename)
    while true
      error("This is actually buggy and needs fixing")
      raw = collect(take(f, provider.batch_size))
      cnt = length(raw)
      if cnt == 0
        # end of file, no more data to see
        return
      end

      data_jl[:] = 0
      for i = 1:provider.batch_size
        vec, gnd = raw[min(i,cnt)]
        data_jl[1:length(vec),i] = vec
        label_jl[i]  = gnd
      end
      mx.copy!(data_nd, data_jl)
      mx.copy!(label_nd, label_jl)
      batch.count = cnt
      produce(batch)
    end
  end

  return Task(_svmlight_iter)
end
