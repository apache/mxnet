#'
#' Convert symbol to Graphviz or visNetwork visualisation.
#'
#' @importFrom magrittr %>%
#' @importFrom stringr str_extract_all
#' @importFrom stringr str_replace_all
#' @importFrom stringr str_replace_na
#' @importFrom stringr str_trim
#' @importFrom jsonlite fromJSON
#' @importFrom DiagrammeR create_graph
#' @importFrom DiagrammeR set_global_graph_attrs 
#' @importFrom DiagrammeR add_global_graph_attrs
#' @importFrom DiagrammeR create_node_df
#' @importFrom DiagrammeR create_edge_df
#' @importFrom DiagrammeR render_graph
#' @importFrom visNetwork visHierarchicalLayout
#'
#' @param symbol a \code{string} representing the symbol of a model.
#' @param shape a \code{numeric} representing the input dimensions to the symbol.
#' @param direction a \code{string} representing the direction of the graph, either TD or LR.
#' @param type a \code{string} representing the rendering engine of the the graph, either graph or vis.
#' @param graph.width.px a \code{numeric} representing the size (width) of the graph. In pixels
#' @param graph.height.px a \code{numeric} representing the size (height) of the graph. In pixels
#'
#' @return a graph object ready to be displayed with the \code{print} function.
#'
#' @export
graph.viz <- function(symbol, shape=NULL, direction="TD", type="graph", graph.width.px=NULL, graph.height.px=NULL){
  
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
      "LinearRegressionOutput"=,
      "MAERegressionOutput"=,
      "SVMOutput"=,
      "LogisticRegressionOutput"=,
      "SoftmaxOutput" = "#b3de69",
      "#fccde5" # default value
    )
  }
  
  get.shape <- function(type) {
    switch(
      EXPR = type,
      "data" = "oval",
      "Pooling" = "oval",
      "Flatten" = "oval",
      "Reshape" = "oval",
      "Concat" = "oval",
      "box" # default value
    )
  }
  
  model_list<- fromJSON(symbol$as.json())
  model_nodes<- model_list$nodes
  model_nodes$id<- 1:nrow(model_nodes)-1
  model_nodes$level<- model_nodes$ID
  
  # extract IDs from string list
  tuple_str <- function(str) sapply(str_extract_all(str, "\\d+"), function(x) paste0(x, collapse="X"))
  
  ### substitute op for heads
  op_id<- sort(unique(model_list$heads[1,]+1))
  op_null<- which(model_nodes$op=="null")
  op_substitute<- intersect(op_id, op_null)
  model_nodes$op[op_substitute]<- model_nodes$name[op_substitute]
  
  model_nodes$color<- apply(model_nodes["op"], 1, get.color)
  model_nodes$shape<- apply(model_nodes["op"], 1, get.shape)
  
  label_paste <- paste0(
    model_nodes$op,
    "\n",
    model_nodes$name,
    "\n",
    model_nodes$attr$num_hidden %>% str_replace_na() %>% str_replace_all(pattern = "NA", ""),
    model_nodes$attr$act_type %>% str_replace_na() %>% str_replace_all(pattern = "NA", ""),
    model_nodes$attr$pool_type %>% str_replace_na() %>% str_replace_all(pattern = "NA", ""),
    model_nodes$attr$kernel %>% tuple_str %>% str_replace_na() %>% str_replace_all(pattern = "NA", ""),
    " / ",
    model_nodes$attr$stride %>% tuple_str %>% str_replace_na() %>% str_replace_all(pattern = "NA", ""),
    ", ",
    model_nodes$attr$num_filter %>% str_replace_na() %>% str_replace_all(pattern = "NA", "")
  ) %>% 
    str_replace_all(pattern = "[^[:alnum:]]+$", "")  %>% 
    str_trim
  
  model_nodes$label<- label_paste
  
  id.to.keep <- model_nodes$id[!model_nodes$op=="null"]
  nodes_df <- model_nodes[model_nodes$id %in% id.to.keep, c("id", "label", "shape", "color")]
  
  ### remapping for DiagrammeR convention
  nodes_df$id<- nodes_df$id
  nodes_df$id_graph<- 1:nrow(nodes_df)
  id_dic<- nodes_df$id_graph
  names(id_dic)<- as.character(nodes_df$id)
  
  edges_id<- model_nodes$id[!sapply(model_nodes$inputs, length)==0 & !model_nodes$op=="null"]
  edges_id<- id_dic[as.character(edges_id)]
  edges<- model_nodes$inputs[!sapply(model_nodes$inputs, length)==0 & !model_nodes$op=="null"]
  edges<- sapply(edges, function(x)intersect(as.numeric(x[, 1]), id.to.keep), simplify = F)
  names(edges)<- edges_id
  
  edges_df<- data.frame(
    from=unlist(edges),
    to=rep(names(edges), time=sapply(edges, length)),
    arrows = "to",
    color="black",
    from_name_output=paste0(model_nodes$name[unlist(edges)+1], "_output"), 
    stringsAsFactors=F)
  edges_df$from<- id_dic[as.character(edges_df$from)]
  
  nodes_df_new<- create_node_df(n = nrow(nodes_df), label=nodes_df$label, shape=nodes_df$shape, type="base", penwidth=2, color=nodes_df$color, style="filled", fillcolor=adjustcolor(nodes_df$color, alpha.f = 1))
  edge_df_new<- create_edge_df(from = edges_df$from, to=edges_df$to, color="black")
  
  if (!is.null(shape)){
    edges_labels_raw<- symbol$get.internals()$infer.shape(list(data=shape))$out.shapes
    if (!is.null(edges_labels_raw)){
      edge_label_str <- function(x) paste0(x, collapse="X")
      edges_labels_raw<- sapply(edges_labels_raw, edge_label_str)
      names(edges_labels_raw)[names(edges_labels_raw)=="data"]<- "data_output"
      edge_df_new$label<- edges_labels_raw[edges_df$from_name_output]
      edge_df_new$rel<- edge_df_new$label
    }
  }
  
  graph<- create_graph(nodes_df = nodes_df_new, edges_df = edge_df_new, directed = T) %>% 
    set_global_graph_attrs("layout", value = "dot", attr_type = "graph") %>% 
    add_global_graph_attrs("rankdir", value = direction, attr_type = "graph")
  
  if (type=="vis"){
    graph_render<- render_graph(graph = graph, output = "visNetwork", width = graph.width.px, height = graph.height.px) %>% visHierarchicalLayout(direction = direction, sortMethod = "directed")
  } else {
    graph_render<- render_graph(graph = graph, output = "graph", width = graph.width.px, height = graph.height.px)
  }

  # graph <-visNetwork(nodes = nodes_df, edges = edges_df, main = graph.title) %>%
  #   visHierarchicalLayout(direction = "UD", sortMethod = "directed")
  
  return(graph_render)
}

globalVariables(c("color", "shape", "label", "id", ".", "op"))