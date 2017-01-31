#'
#' Convert symbol to dot object for visualization purpose.
#'
#' @importFrom magrittr %>%
#' @importFrom stringr str_extract_all
#' @importFrom stringr str_replace_all
#' @importFrom stringr str_replace_na
#' @importFrom stringr str_trim
#' @importFrom jsonlite fromJSON
#' @importFrom visNetwork visNetwork
#' @importFrom visNetwork visHierarchicalLayout
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
graph.viz <- function(symbol, shape=NULL, graph.title = "Computation graph", graph.title.font.name = "Helvetica", graph.title.font.size = 30, graph.width.px = 500, graph.height.px = 500){
  
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
      "Pooling" = "oval",
      "Flatten" = "triangleDown",
      "Reshape" = "triangleDown",
      "Concat" = "triangleDown",
      "box" # default value
    )
  }
  
  model_list<- fromJSON(symbol$as.json())
  model_nodes<- model_list$nodes
  model_nodes$id<- 1:nrow(model_nodes)-1
  model_nodes$level<- model_nodes$ID
  
  # extract IDs from string list
  tuple_str <- function(str) sapply(str_extract_all(str, "\\d+"), function(x) paste0(x, collapse="X"))
  
  label_paste <- paste0(
    model_nodes$name,
    "\n",
    model_nodes$attr$num_hidden %>% str_replace_na() %>% str_replace_all(pattern = "NA", ""),
    model_nodes$attr$pool_type %>% str_replace_na() %>% str_replace_all(pattern = "NA", ""),
    model_nodes$attr$kernel %>% tuple_str %>% str_replace_na() %>% str_replace_all(pattern = "NA", ""),
    " / ",
    model_nodes$attr$stride %>% tuple_str %>% str_replace_na() %>% str_replace_all(pattern = "NA", ""),
    ", ",
    model_nodes$attr$num_filter %>% str_replace_na() %>% str_replace_all(pattern = "NA", "")
  ) %>% 
    str_replace_all(pattern = "[^[:alnum:]]+$", "")  %>% 
    str_trim #%>% 
  #str_replace(pattern = "\\s+", " ")  %>% 
  #str_replace(pattern = "[^[:alnum:]]+$", "")
  
  
  model_nodes$op[model_list$heads[1,]+1]<- model_nodes$name[model_list$heads[1,]+1]
  model_nodes$color<- apply(model_nodes["op"], 1, get.color)
  model_nodes$shape<- apply(model_nodes["op"], 1, get.shape)
  model_nodes$label<- label_paste
  
  id.to.keep <- model_nodes$id[!model_nodes$op=="null"]
  nodes_df <- model_nodes[model_nodes$id %in% id.to.keep, c("id", "label", "shape", "color")]
  
  edges_id<- model_nodes$id[!sapply(model_nodes$inputs, length)==0 & !model_nodes$op=="null"]
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
  
  if (!is.null(shape)){
    edges_labels_raw<- symbol$get.internals()$infer.shape(list(data=shape))$out.shapes
    edge_label_str <- function(x) paste0(x, collapse="X")
    edges_labels_raw<- sapply(edges_labels_raw, edge_label_str)
    edges_df$label<- edges_labels_raw[edges_df$from_name_output]
  }
  
  graph <-visNetwork(nodes = nodes_df, edges = edges_df, main = graph.title) %>%
    visHierarchicalLayout(direction = "UD", sortMethod = "directed")
  
  return(graph)
}

globalVariables(c("color", "shape", "label", "id", ".", "op"))
