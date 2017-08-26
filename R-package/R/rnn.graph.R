# 
#' Generate a RNN symbolic model
#' 
#' @param config Either seq-to-one or one-to-one
#' @param cell.type Type of RNN cell: either gru or lstm
#' @param num.rnn.layer int, number of stacked layers
#' @param num.hidden int, size of the state in each RNN layer
#' @param num.embed  int, dimension of the embedding vectors
#' @param num.label int, number of categories in labels
#' @param input.size int, number of levels in the data
#' @param dropout
#' 
#' @export
rnn.graph <- function(num.rnn.layer, 
                      input.size,
                      num.embed, 
                      num.hidden,
                      num.label,
                      dropout = 0,
                      ignore_label = -1,
                      config,
                      cell.type,
                      masking = T,
                      output_last_state = F) {
  
  # define input arguments
  label <- mx.symbol.Variable("label")
  data <- mx.symbol.Variable("data")
  seq.mask <- mx.symbol.Variable("seq.mask")
  
  embed.weight <- mx.symbol.Variable("embed.weight")
  rnn.params.weight <- mx.symbol.Variable("rnn.params.weight")
  
  rnn.state <- mx.symbol.Variable("rnn.state")

  if (cell.type == "lstm") {
    rnn.state.cell <- mx.symbol.Variable("rnn.state.cell")
  }
  
  cls.weight <- mx.symbol.Variable("cls.weight")
  cls.bias <- mx.symbol.Variable("cls.bias")
  
  data <- mx.symbol.transpose(data=data)  
  embed <- mx.symbol.Embedding(data=data, input_dim=input.size,
                               weight=embed.weight, output_dim=num.embed, name="embed")
  
  # RNN cells
  if (cell.type == "lstm") {
    rnn <- mx.symbol.RNN(data=embed, state=rnn.state, state_cell = rnn.state.cell, parameters=rnn.params.weight, state.size=num.hidden, num.layers=num.rnn.layer, bidirectional=F, mode=cell.type, state.outputs=output_last_state, p=dropout, name=paste(cell.type, num.rnn.layer, "layer", sep="_"))
    
  } else {
    rnn <- mx.symbol.RNN(data=embed, state=rnn.state, parameters=rnn.params.weight, state.size=num.hidden, num.layers=num.rnn.layer, bidirectional=F, mode=cell.type, state.outputs=output_last_state, p=dropout, name=paste(cell.type, num.rnn.layer, "layer", sep="_"))
  }
  
  # Decode
  if (config=="seq-to-one") {
    
    if (masking) mask <- mx.symbol.SequenceLast(data=rnn[[1]], use.sequence.length = T, sequence_length = seq.mask, name = "mask") else
      mask <- mx.symbol.SequenceLast(data=rnn[[1]], use.sequence.length = F, name = "mask")
    
    fc <- mx.symbol.FullyConnected(data=mask,
                                   weight=cls.weight,
                                   bias=cls.bias,
                                   num.hidden=num.label,
                                   name = "decode")
    
    loss <- mx.symbol.SoftmaxOutput(data=fc, label=label, use_ignore = !ignore_label == -1, ignore_label = ignore_label, name = "loss")
    
  } else if (config=="one-to-one"){
    
    if (masking) mask <- mx.symbol.SequenceMask(data = rnn[[1]], use.sequence.length = T, sequence_length = seq.mask, value = 0, name = "mask") else
    mask <- mx.symbol.identity(data = rnn[[1]], name = "mask")
    
    reshape = mx.symbol.transpose(mask)
    flatten = mx.symbol.flatten(reshape)
    transpose = mx.symbol.transpose(flatten)
    
    decode <- mx.symbol.FullyConnected(data=transpose,
                                   weight=cls.weight,
                                   bias=cls.bias,
                                   num.hidden=num.label,
                                   name = "decode")
    
    label <- mx.symbol.reshape(data=label, shape=c(-1))
    loss <- mx.symbol.SoftmaxOutput(data=decode, label=label, use_ignore = !ignore_label == -1, ignore_label = ignore_label, name = "loss")
    
  }
  
  if (output_last_state){
    # WIP for one-to-one
    # if (cell.type == "lstm") group <- mx.symbol.Group(rnn[[2]], rnn[[3]], loss) else 
    #   group <- mx.symbol.Group(rnn[[2]], loss)
    # return(group)
    return(loss)
  } else return(loss)
}

