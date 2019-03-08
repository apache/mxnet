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

# An explicitly unrolled LSTM with fixed sequence length.
using MXNet

#--LSTMState
struct LSTMState
  c :: mx.SymbolicNode
  h :: mx.SymbolicNode
end
#--/LSTMState

#--LSTMParam
struct LSTMParam
  i2h_W :: mx.SymbolicNode
  h2h_W :: mx.SymbolicNode
  i2h_b :: mx.SymbolicNode
  h2h_b :: mx.SymbolicNode
end
#--/LSTMParam

#--lstm_cell
function lstm_cell(data::mx.SymbolicNode, prev_state::LSTMState, param::LSTMParam;
                   num_hidden::Int=512, dropout::Real=0, name::Symbol=gensym())

  if dropout > 0
    data = mx.Dropout(data, p=dropout)
  end

  i2h = mx.FullyConnected(data, weight=param.i2h_W, bias=param.i2h_b,
                          num_hidden=4num_hidden, name=Symbol(name, "_i2h"))
  h2h = mx.FullyConnected(prev_state.h, weight=param.h2h_W, bias=param.h2h_b,
                          num_hidden=4num_hidden, name=Symbol(name, "_h2h"))

  gates = mx.SliceChannel(i2h + h2h, num_outputs=4, name=Symbol(name, "_gates"))

  in_gate     = mx.Activation(gates[1], act_type=:sigmoid)
  in_trans    = mx.Activation(gates[2], act_type=:tanh)
  forget_gate = mx.Activation(gates[3], act_type=:sigmoid)
  out_gate    = mx.Activation(gates[4], act_type=:sigmoid)

  next_c = (forget_gate .* prev_state.c) + (in_gate .* in_trans)
  next_h = out_gate .* mx.Activation(next_c, act_type=:tanh)

  return LSTMState(next_c, next_h)
end
#--/lstm_cell

#--LSTM-part1
function LSTM(n_layer::Int, seq_len::Int, dim_hidden::Int, dim_embed::Int, n_class::Int;
              dropout::Real=0, name::Symbol=gensym(), output_states::Bool=false)

  # placeholder nodes for all parameters
  embed_W = mx.Variable(Symbol(name, "_embed_weight"))
  pred_W  = mx.Variable(Symbol(name, "_pred_weight"))
  pred_b  = mx.Variable(Symbol(name, "_pred_bias"))

  layer_param_states = map(1:n_layer) do i
    param = LSTMParam(mx.Variable(Symbol(name, "_l$(i)_i2h_weight")),
                      mx.Variable(Symbol(name, "_l$(i)_h2h_weight")),
                      mx.Variable(Symbol(name, "_l$(i)_i2h_bias")),
                      mx.Variable(Symbol(name, "_l$(i)_h2h_bias")))
    state = LSTMState(mx.Variable(Symbol(name, "_l$(i)_init_c")),
                      mx.Variable(Symbol(name, "_l$(i)_init_h")))
    (param, state)
  end
  #...
  #--/LSTM-part1

  #--LSTM-part2
  # now unroll over time
  outputs = mx.SymbolicNode[]
  for t = 1:seq_len
    data   = mx.Variable(Symbol(name, "_data_$t"))
    label  = mx.Variable(Symbol(name, "_label_$t"))
    hidden = mx.FullyConnected(data, weight=embed_W, num_hidden=dim_embed,
                               no_bias=true, name=Symbol(name, "_embed_$t"))

    # stack LSTM cells
    for i = 1:n_layer
      l_param, l_state = layer_param_states[i]
      dp = i == 1 ? 0 : dropout # don't do dropout for data
      next_state = lstm_cell(hidden, l_state, l_param, num_hidden=dim_hidden, dropout=dp,
                             name=Symbol(name, "_lstm_$t"))
      hidden = next_state.h
      layer_param_states[i] = (l_param, next_state)
    end

    # prediction / decoder
    if dropout > 0
      hidden = mx.Dropout(hidden, p=dropout)
    end
    pred = mx.FullyConnected(hidden, weight=pred_W, bias=pred_b, num_hidden=n_class,
                             name=Symbol(name, "_pred_$t"))
    smax = mx.SoftmaxOutput(pred, label, name=Symbol(name, "_softmax_$t"))
    push!(outputs, smax)
  end
  #...
  #--/LSTM-part2

  #--LSTM-part3
  # append block-gradient nodes to the final states
  for i = 1:n_layer
    l_param, l_state = layer_param_states[i]
    final_state = LSTMState(mx.BlockGrad(l_state.c, name=Symbol(name, "_l$(i)_last_c")),
                            mx.BlockGrad(l_state.h, name=Symbol(name, "_l$(i)_last_h")))
    layer_param_states[i] = (l_param, final_state)
  end

  # now group all outputs together
  if output_states
    outputs = outputs ∪ [x[2].c for x in layer_param_states] ∪
                        [x[2].h for x in layer_param_states]
  end
  return mx.Group(outputs...)
end
#--/LSTM-part3


# Negative Log-likelihood
mutable struct NLL <: mx.AbstractEvalMetric
  nll_sum  :: Float64
  n_sample :: Int

  NLL() = new(0.0, 0)
end

function mx.update!(metric::NLL, labels::Vector{<:mx.NDArray}, preds::Vector{<:mx.NDArray})
  @assert length(labels) == length(preds)
  nll = 0.0
  for (label, pred) in zip(labels, preds)
    @mx.nd_as_jl ro=(label, pred) begin
      nll -= sum(
        log.(
          max.(
            getindex.(
            (pred,),
            round.(Int,label .+ 1),
            1:length(label)),
          1e-20)
        )
      )
    end
  end

  nll = nll / length(labels)
  metric.nll_sum += nll
  metric.n_sample += length(labels[1])
end

function mx.get(metric :: NLL)
  nll  = metric.nll_sum / metric.n_sample
  perp = exp(nll)
  return [(:NLL, nll), (:perplexity, perp)]
end

function mx.reset!(metric :: NLL)
  metric.nll_sum  = 0.0
  metric.n_sample = 0
end
