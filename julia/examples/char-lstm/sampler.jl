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

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "lstm.jl"))
include(joinpath(@__DIR__, "seq-data.jl"))

using StatsBase
using MXNet

# load vocabulary
vocab   = build_vocabulary(INPUT_FILE, VOCAB_FILE)
n_class = length(vocab)

# prepare data provider
jl_data = Pair[(Symbol(NAME, "_data_$t") => zeros(mx.MX_float, (length(vocab), BATCH_SIZE_SMP)))
               for t = 1:1]
jl_c    = Pair[(Symbol(NAME, "_l$(l)_init_c") => zeros(mx.MX_float, (DIM_HIDDEN, BATCH_SIZE_SMP)))
               for l = 1:LSTM_N_LAYER]
jl_h    = Pair[(Symbol(NAME, "_l$(l)_init_h") => zeros(mx.MX_float, (DIM_HIDDEN, BATCH_SIZE_SMP)))
               for l = 1:LSTM_N_LAYER]

# the first input in the sequence
jl_data_start = jl_data[1].second
jl_data_start[char_idx(vocab, SAMPLE_START),:] = 1

# define a LSTM with sequence length 1, also output states so that we could manually copy the states
# when sampling the next char
lstm  = LSTM(LSTM_N_LAYER, 1, DIM_HIDDEN, DIM_EMBED, n_class, name=NAME, output_states=true)
model = mx.FeedForward(lstm, context=mx.cpu())

# load parameters from traind LSTM, though the sequence length is different, since the weights are shared
# over time, this should be compatible.
model = mx.load_checkpoint(model, CKPOINT_PREFIX, N_EPOCH, allow_different_arch=true)

# prepare outputs
Base.zero(::Type{Char}) = Char(0)
output_samples = zeros(Char, (SAMPLE_LENGTH, BATCH_SIZE_SMP))
output_samples[1, :] = SAMPLE_START

# build inverse vocabulary for convenience
inv_vocab = Dict(v => k for (k,v) in vocab)

# do prediction and sampling step by step
for t = 2:SAMPLE_LENGTH-1
  data    = mx.ArrayDataProvider(jl_data ∪ jl_c ∪ jl_h)
  preds   = mx.predict(model, data)

  # the first output is prediction
  outputs = preds[1]

  # do sampling and init the next inputs
  jl_data_start[:] = 0
  for i = 1:BATCH_SIZE_SMP
    prob = WeightVec(outputs[:, i])
    k    = sample(prob)
    output_samples[t, i] = inv_vocab[k]
    jl_data_start[k, i]  = 1
  end

  # copy the states over
  for l = 1:LSTM_N_LAYER
    copy!(jl_c[l][2], preds[1+l])
    copy!(jl_h[l][2], preds[1+LSTM_N_LAYER+l])
  end
end

output_texts = [join(output_samples[:,i]) for i = 1:BATCH_SIZE_SMP]
output_texts = [replace(x, UNKNOWN_CHAR, '?') for x in output_texts]

for (i, text) in enumerate(output_texts)
  println("## Sample $i")
  println(text)
  println()
end
