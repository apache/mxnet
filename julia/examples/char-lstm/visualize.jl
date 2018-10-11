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

include(joinpath(dirname(@__FILE__), "config.jl"))
include(joinpath(dirname(@__FILE__), "lstm.jl"))

using MXNet

vis_n_layer = 1
vis_seq_len = 2
vis_n_class = 128

lstm  = LSTM(vis_n_layer, vis_seq_len, DIM_HIDDEN, DIM_EMBED, vis_n_class, name=NAME, output_states=true)

open("visualize.dot", "w") do io
  println(io, mx.to_graphviz(lstm))
end
run(pipeline(`dot -Tsvg visualize.dot`, stdout="visualize.svg"))
