#' 
#' Convert symbol to dot object for visualization purpose.
#' 
#' @importFrom magrittr %>%
#' @importFrom stringr str_extract_all
#' @importFrom magrittr %>%
#' @importFrom data.table data.table
#' @importFrom data.table as.data.table
#' @importFrom data.table :=
#' @importFrom data.table setkey
#' @importFrom jsonlite fromJSON
#' @importFrom DiagrammeR create_nodes
#' @importFrom DiagrammeR create_graph
#' @importFrom DiagrammeR create_edges
#' @importFrom DiagrammeR combine_edges
#' @importFrom DiagrammeR render_graph
#' 
#' @param model a \code{string} representing the path to a file containing the \code{JSon} of a model dump or the actual model dump.
#' @param graph.title a \code{string} displayed on top of the viz.
#' @param graph.title.font.name a \code{string} representing the font to use for the title.
#' @param  graph.title.font.size a \code{numeric} representing the size of the font to use for the title.
#' @param graph.width.px a \code{numeric} representing the size (width) of the graph. In pixels
#' @param graph.height.px a \code{numeric} representing the size (height) of the graph. In pixels
#'     
#' @return a graph object ready to be displayed with the \code{print} function.
#'
#' @export
graph.viz <- function(model, graph.title = "Computation graph", graph.title.font.name = "Helvetica", graph.title.font.size = 30, graph.width.px = 500, graph.height.px = 500){
  # generate color code for each type of node.
  get.color <- function(type) {
    switch(
      EXPR = type,
      "data" = "#8dd3c7",
      "FullyConnected" = ,
      "Convolution" = "#fb8072",
      "LeakyReLU" = ,
      "Activation" = "#ffffb3",
      "BatchNorm" = "#bebada",
      "Pooling" = "#80b1d3",
      "Flatten" = ,
      "Reshape" = ,
      "Concat" = "#fdb462",
      "Softmax" = "#b3de69",
      "#fccde5" # default value
    )
  }
  
  get.shape <- function(type) {
    switch(
      EXPR = type,
      "data" = "star",
      #     "FullyConnected" = ,
      #     "Convolution" = "#fb8072",
      #    "LeakyReLU" = ,
      #    "Activation" = "diamond",
      #     "BatchNorm" = "#bebada",
      "Pooling" = "oval",
      "Flatten" = ,
      "Reshape" = ,
      "Concat" = "invtriangle",
      #     "Softmax" = "#b3de69",
      "box" # default value
    )
  }
  
  # extract IDs from string list
  str2tuple <- function(str) str_extract_all(str, "\\d+") %>% unlist %>% as.numeric
  
  # generate text content for each node.
  get.label <- function(type, mat.row) {
    switch(
      EXPR = type,
      "FullyConnected" = mat.row[,param.num_hidden] %>% paste("FullyConnected", ., sep = "\n"),
      "Convolution" = {
        kernel.parameters <- mat.row[,param.kernel] %>% str2tuple
        stride.parameters <- mat.row[,param.stride] %>% str2tuple
        num_filter.parameters <- mat.row[,param.num_filter] %>% str2tuple
        paste0("Convolution\n", kernel.parameters[1], "x", kernel.parameters[2],
               "/", stride.parameters[1], ", ", num_filter.parameters)
      }, 
      "LeakyReLU" = ,
      "Activation" = mat.row[,param.act_type] %>% paste0(type, "\n", .),
      "Pooling" = {
        pool_type.parameters <- mat.row[,param.pool_type] %>% str2tuple
        kernel.parameters <- mat.row[,param.kernel] %>% str2tuple
        stride.parameters <- mat.row[,param.stride] %>% str2tuple
        paste0("Pooling\n", pool_type.parameters, "\n", kernel.parameters[1], "x", 
               kernel.parameters[2], "/", stride.parameters[1])
      },
      type # default value
    )
  }
  
  mx.model.json <- fromJSON(model, flatten = T)
  mx.model.nodes <- mx.model.json$nodes %>% as.data.table
  mx.model.nodes[,id:= .I - 1]
  setkey(mx.model.nodes, id)
  mx.model.json$heads[1,] %>% {mx.model.nodes[id %in% .,op:=name]} # add nodes from heads (mainly data node)
  mx.model.nodes[,color:= get.color(op), by = id] # by=id to have an execution row per row
  mx.model.nodes[,shape:= get.shape(op), by = id] # by=id to have an execution row per row
  mx.model.nodes[,label:= get.label(op, .SD), by = id] # by=id to have an execution row per row
  
  nodes.to.keep <-
    mx.model.nodes[op != "null",id] %>% unique %>% sort
  nodes.to.remove <-
    mx.model.nodes[,id] %>% unique %>% setdiff(nodes.to.keep) %>% sort
  
  nodes <-
    create_nodes(
      nodes = mx.model.nodes[id %in% nodes.to.keep, id],
      label = mx.model.nodes[id %in% nodes.to.keep, label],
      type = "lower",
      style = "filled",
      fillcolor  = mx.model.nodes[id %in% nodes.to.keep, color],
      shape = mx.model.nodes[id %in% nodes.to.keep, shape],
      data = mx.model.nodes[id %in% nodes.to.keep, id],
      #fixedsize = TRUE,
      width = "1.3",
      height = "0.8034"
    )
  
  mx.model.nodes[,has.connection:= sapply(inputs, function(x)
    length(x) > 0)]
  
  nodes.to.insert <-
    mx.model.nodes[id %in% nodes.to.keep &
                     has.connection == T, .(id, inputs)]
  
  edges <- NULL
  for (i in 1:nrow(nodes.to.insert)) {
    current.id <- nodes.to.insert[i, id]
    origin <-
      nodes.to.insert[i, inputs][[1]][,1] %>% setdiff(nodes.to.remove) %>% unique
    destination <- rep(current.id, length(origin))
    edges.temp <- create_edges(from = origin,
                               to = destination,
                               relationship = "leading_to")
    if (is.null(edges))
      edges <- edges.temp
    else
      edges <- combine_edges(edges.temp, edges)
  }
  
  graph <-
    create_graph(
      nodes_df = nodes,
      edges_df = edges,
      directed = TRUE,
      # node_attrs = c("fontname = Helvetica"),
      graph_attrs = paste0("label = \"", graph.title, "\"") %>% c(paste0("fontname = ", graph.title.font.name)) %>% c(paste0("fontsize = ", graph.title.font.size)) %>% c("labelloc = t"),
      # node_attrs = "fontname = Helvetica",
      edge_attrs = c("color = gray20", "arrowsize = 0.8", "arrowhead = vee")
    )
  
  return(render_graph(graph, width = graph.width.px, height = graph.height.px))
}

globalVariables(c("color", "shape", "label", "id", ".", "op"))
