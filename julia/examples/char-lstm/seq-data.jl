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

# Simple data provider that load text
using Iterators
using MXNet

function build_vocabulary(corpus_fn::AbstractString, vocab_fn::AbstractString; max_vocab=10000)
  if isfile(vocab_fn)
    info("Vocabulary already exists, reusing $vocab_fn...")
    vocab = Dict{Char,Int}(w => i for (i,w) in enumerate(readstring(vocab_fn)))
  else
    # count symbol frequency
    dict = Dict{Char,Int}()
    open(corpus_fn) do io
      for line in eachline(io)
        for c in line
          dict[c] = get(dict, c, 0) + 1
        end
      end
    end

    vocab = sort(collect(dict), by=x->-x.second)
    vocab = vocab[1:min(max_vocab,length(vocab))]
    open(vocab_fn, "w") do io
      for x in vocab
        print(io, x.first)
      end
    end

    vocab = Dict(x.first => i for (i,x) in enumerate(vocab))
  end
  vocab[UNKNOWN_CHAR] = length(vocab)
  return vocab
end

#--CharSeqProvider
mutable struct CharSeqProvider <: mx.AbstractDataProvider
  text       :: AbstractString
  batch_size :: Int
  seq_len    :: Int
  vocab      :: Dict{Char,Int}

  prefix     :: Symbol
  n_layer    :: Int
  dim_hidden :: Int
end
#--/CharSeqProvider

function mx.get_batch_size(p :: CharSeqProvider)
  p.batch_size
end

#--provide
function mx.provide_data(p :: CharSeqProvider)
  [(Symbol(p.prefix, "_data_$t"), (length(p.vocab), p.batch_size)) for t = 1:p.seq_len] ∪
  [(Symbol(p.prefix, "_l$(l)_init_c"), (p.dim_hidden, p.batch_size)) for l=1:p.n_layer] ∪
  [(Symbol(p.prefix, "_l$(l)_init_h"), (p.dim_hidden, p.batch_size)) for l=1:p.n_layer]
end
function mx.provide_label(p :: CharSeqProvider)
  [(Symbol(p.prefix, "_label_$t"), (p.batch_size,)) for t = 1:p.seq_len]
end
#--/provide

#--eachbatch-part1
function mx.eachbatch(p::CharSeqProvider)
  data_all  = [mx.zeros(shape) for (name, shape) in mx.provide_data(p)]
  label_all = [mx.zeros(shape) for (name, shape) in mx.provide_label(p)]

  data_jl = [copy(x) for x in data_all]
  label_jl= [copy(x) for x in label_all]

  batch = mx.DataBatch(data_all, label_all, p.batch_size)
  #...
  #--/eachbatch-part1

  #--eachbatch-part2
  #...
  function _text_iter(c::Channel)
    text = p.text

    n_batch = floor(Int, length(text) / p.batch_size / p.seq_len)
    text = text[1:n_batch*p.batch_size*p.seq_len] # discard tailing
    idx_all = 1:length(text)

    for idx_batch in partition(idx_all, p.batch_size*p.seq_len)
      for i = 1:p.seq_len
        data_jl[i][:] = 0
        label_jl[i][:] = 0
      end

      for (i, idx_seq) in enumerate(partition(idx_batch, p.seq_len))
        for (j, idx) in enumerate(idx_seq)
          c_this = text[idx]
          c_next = idx == length(text) ? UNKNOWN_CHAR : text[idx+1]
          data_jl[j][char_idx(vocab,c_this),i] = 1
          label_jl[j][i] = char_idx(vocab,c_next)-1
        end
      end

      for i = 1:p.seq_len
        copy!(data_all[i], data_jl[i])
        copy!(label_all[i], label_jl[i])
      end

      put!(c, batch)
    end
  end

  return Channel(_text_iter)
end
#--/eachbatch-part2

# helper function to convert a char into index in vocabulary
function char_idx(vocab :: Dict{Char,Int}, c :: Char)
  if haskey(vocab, c)
    vocab[c]
  else
    vocab[UNKNOWN_CHAR]
  end
end

