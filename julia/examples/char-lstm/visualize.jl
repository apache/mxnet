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
