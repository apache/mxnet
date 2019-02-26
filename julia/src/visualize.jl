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

import JSON

"""
    to_graphviz(network)

* `network::SymbolicNode`: the network to visualize.
* `title::AbstractString:` keyword argument, default "Network Visualization",
          the title of the GraphViz graph.
* `input_shapes`: keyword argument, default `nothing`. If provided,
          will run shape inference and plot with the shape information. Should
          be either a dictionary of name-shape mapping or an array of shapes.

Returns the graph description in GraphViz `dot` language.
"""
function to_graphviz(network :: SymbolicNode; title="Network Visualization", input_shapes=nothing)
  if !isa(input_shapes, Cvoid)
    internals = get_internals(network)
    if isa(input_shapes, Dict)
      _, out_shapes, _ = infer_shape(internals; input_shapes...)
    else
      _, out_shapes, _ = infer_shape(internals, input_shapes...)
    end
    @assert(!isa(out_shapes, Cvoid), "Failed to do shape inference, input shapes are incomplete")
    shape_dict = Dict(zip(list_outputs(internals), out_shapes))
    draw_shape = true
  else
    draw_shape = false
  end

  conf = JSON.parse(to_json(network))
  nodes = conf["nodes"]
  heads = unique([x[1]+1 for x in conf["heads"]])
  node_attr = Dict(:shape => :box, :fixedsize => true, :width => 1.3,
                   :height => 0.8034, :style => (:rounded, :filled), :penwidth => 2)
  io = IOBuffer()
  println(io, "digraph $(_simple_escape(title)) {")
  println(io, "node [fontsize=10];")
  println(io, "edge [fontsize=10];")

  # color map
  fillcolors = ("#8dd3c7", "#fb8072", "#ffffb3", "#bebada", "#80b1d3",
                "#fdb462", "#b3de69", "#fccde5")
  edgecolors = ("#245b51", "#941305", "#999900", "#3b3564", "#275372",
                "#975102", "#597d1c", "#90094e")

  # make nodes
  for i = 1:length(nodes)
    node  = nodes[i]
    op    = node["op"]
    name  = node["name"]
    attr  = deepcopy(node_attr)
    label = op

    # Up to 0.11.0 version of mxnet additional info was stored in
    # node["attr"]. Staring from 0.12 `attr` was changed to `attrs`.
    # See: https://github.com/dmlc/nnvm/pull/152
    if haskey(node, "attrs")
      node_info = node["attrs"]
    elseif haskey(node, "attr")
      node_info = node["attr"]
    end

    if op == "null"
      if i ∈ heads
        # heads are output nodes
        label = node["name"]
        colorkey = 1
      else
        # otherwise, input nodes, might be data, label or parameters
        continue
      end
    elseif op == "Convolution"
      if haskey(node_info,"stride")
        stride_info=_extract_shape(node_info["stride"])
      else
        stride_info="1"
      end

      label = format("Convolution\nkernel={1}\nstride={2}\nn-filter={3}",
                     _extract_shape(node_info["kernel"]),
                     stride_info,
                     node_info["num_filter"])
      colorkey = 2
    elseif op == "FullyConnected"
      label = format("FullyConnected\nnum-hidden={1}", node_info["num_hidden"])
      colorkey = 2
    elseif op == "Activation"
      label = format("Activation\nact-type={1}", node_info["act_type"])
      colorkey = 3
    elseif op == "BatchNorm"
      colorkey = 4
    elseif op == "Pooling"
      if haskey(node_info,"stride")
        stride_info=_extract_shape(node_info["stride"])
      else
        stride_info="1"
      end
      label = format("Pooling\ntype={1}\nkernel={2}\nstride={3}",
                     node_info["pool_type"],
                     _extract_shape(node_info["kernel"]),
                     stride_info)
      colorkey = 5
    elseif op ∈ ("Concat", "Flatten", "Reshape")
      colorkey = 6
    elseif endswith(op, "Output") || op == "BlockGrad"
      colorkey = 7
    else
      colorkey = 8
    end

    if op != "null"
      label = "$name\n$label"
    end
    attr[:fillcolor] = fillcolors[colorkey]
    attr[:color]     = edgecolors[colorkey]
    attr[:label]     = label
    _format_graphviz_node(io, name, attr)
  end

  # add edges
  for i = 1:length(nodes)
    node  = nodes[i]
    op    = node["op"]
    name  = node["name"]
    if op == "null"
      continue
    end
    inputs = node["inputs"]
    for item in inputs
      input_node = nodes[item[1]+1]
      input_name = input_node["name"]
      if input_node["op"] != "null" || (item[1]+1) ∈ heads
        attr = Dict(:dir => :back, :arrowtail => :open, :color => "#737373")
        if draw_shape
          if input_node["op"] != "null"
            key   = Symbol(input_name, "_output")
            shape = shape_dict[key][1:end-1]
          else
            key   = Symbol(input_name)
            shape = shape_dict[key][1:end-1]
          end
          label = "(" * join([string(x) for x in shape], ",") * ")"
          attr[:label] = label
        end
        _format_graphviz_edge(io, name, input_name, attr)
      end
    end
  end
  println(io, "}")

  return String(take!(io))
end

function _format_graphviz_attr(io::IOBuffer, attrs)
  label = get(attrs, :label, nothing)
  if isa(label, Cvoid)
    print(io, " [")
  else
    print(io, " [label=$(_simple_escape(label)),")
  end
  first_attr = true
  for (k,v) in attrs
    if k != :label
      if !first_attr
        print(io, ",")
      end
      first_attr = false

      if isa(v, AbstractString) && v[1] == '#'
        # color
        v = _simple_escape(v)
      elseif isa(v, Tuple)
        v = _simple_escape(join([string(x) for x in v], ","))
      end
      print(io, "$k=$v")
    end
  end
  println(io, "];")
end
function _simple_escape(str)
  str = replace(string(str), r"\n" =>  "\\n")
  return "\"$str\""
end
function _format_graphviz_node(io::IOBuffer, name::AbstractString, attrs)
  print(io, "$(_simple_escape(name)) ")
  _format_graphviz_attr(io, attrs)
end
function _format_graphviz_edge(io::IOBuffer, head, tail, attrs)
  print(io, """$(_simple_escape(head)) -> $(_simple_escape(tail)) """)
  _format_graphviz_attr(io, attrs)
end
function _extract_shape(str :: AbstractString)
  shape = matchall(r"\d+", str)
  shape = reverse(shape) # JSON in libmxnet has reversed shape (column vs row majoring)
  return "(" * join(shape, ",") * ")"
end
