import JSON

#=doc
Network Visualization
=====================
=#

#=doc
.. function:: to_graphviz(network)

   :param SymbolicNode network: the network to visualize.
   :param AbstractString title: keyword argument, default "Network Visualization",
          the title of the GraphViz graph.
   :param input_shapes: keyword argument, default ``nothing``. If provided,
          will run shape inference and plot with the shape information. Should
          be either a dictionary of name-shape mapping or an array of shapes.
   :return: the graph description in GraphViz ``dot`` language.
=#
function to_graphviz(network :: SymbolicNode; title="Network Visualization", input_shapes=nothing)
  if !isa(input_shapes, Void)
    internals = get_internals(network)
    if isa(input_shapes, Dict)
      _, out_shapes, _ = infer_shape(; input_shapes...)
    else
      _, out_shapes, _ = infer_shape(input_shapes...)
    end
    @assert(!isa(out_shapes, Void), "Failed to do shape inference, input shapes are incomplete")
    shapes = Dict(zip(list_outputs(internals), out_shapes))
    draw_shape = true
  else
    draw_shape = false
  end

  conf = JSON.parse(to_json(network))
  nodes = conf["nodes"]
  heads = unique(conf["heads"][1]+1)
  node_attr = Dict(:shape => :box, :fixedsize => true, :width => 1.3,
                   :height => 0.8034, :style => :filled)
  io = IOBuffer()
  println(io, "digraph $(_simple_escape(title)) {")
  println(io, "node [fontsize=10];")

  # color map
  cm = ("#8dd3c7", "#fb8072", "#ffffb3", "#bebada", "#80b1d3",
        "#fdb462", "#b3de69", "#fccde5")

  # make nodes
  for i = 1:length(nodes)
    node  = nodes[i]
    op    = node["op"]
    name  = node["name"]
    attr  = deepcopy(node_attr)
    label = op

    if op == "null"
      if i ∈ heads
        label = node["name"]
        attr[:fillcolor] = cm[1]
      else
        continue
      end
    elseif op == "Convolution"
      label = format("Convolution\nkernel={1},stride={2},n-filter={3}",
                     _extract_shape(node["param"]["kernel"]),
                     _extract_shape(node["param"]["stride"]),
                     node["param"]["num_filter"])
      attr[:fillcolor] = cm[2]
    elseif op == "FullyConnected"
      label = format("FullyConnected\nnum-hidden={1}", node["param"]["num_hidden"])
      attr[:fillcolor] = cm[2]
      # TODO: add more
    else
      attr[:fillcolor] = cm[8]
    end

    attr[:label] = label
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
        attr = Dict(:dir => :back, :arrowtail => :open)
        if draw_shape
          if input_node["op"] != "null"
            key   = symbol(input_name, "_output")
            shape = shape_dict[key][1:end-1]
          else
            key   = symbol(input_name)
            shape = shape_dict[key][1:end-1]
          end
          label = "(" * join([string(x) for x in shape], ",") * ")"
          attr[:label] = label
        end
        _format_graphviz_edge(io, input_name, name, attr)
      end
    end
  end
  println(io, "}")

  return takebuf_string(io)
end

function _format_graphviz_attr(io::IOBuffer, attrs)
  label = get(attrs, :label, nothing)
  if isa(label, Void)
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
      end
      print(io, "$k=$v")
    end
  end
  println(io, "];")
end
function _simple_escape(str :: AbstractString)
  str = replace(str, r"\n", "\\n")
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
