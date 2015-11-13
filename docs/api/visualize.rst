
Network Visualization
=====================




.. function:: to_graphviz(network)

   :param SymbolicNode network: the network to visualize.
   :param AbstractString title: keyword argument, default "Network Visualization",
          the title of the GraphViz graph.
   :param input_shapes: keyword argument, default ``nothing``. If provided,
          will run shape inference and plot with the shape information. Should
          be either a dictionary of name-shape mapping or an array of shapes.
   :return: the graph description in GraphViz ``dot`` language.



