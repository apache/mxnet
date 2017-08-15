library(mxnet)

# RNN graph design
rnn.graph <- function(num.rnn.layer, 
                      input.size,
                      num.embed, 
                      num.hidden,
                      num.label,
                      dropout = 0,
                      ignore_label = 0,
                      init.state = NULL,
                      config,
                      cell.type="gru",
                      masking = T,
                      output_last_state = F) {
  
  # define input arguments
  label <- mx.symbol.Variable("label")
  data <- mx.symbol.Variable("data")
  seq.mask <- mx.symbol.Variable("seq.mask")
  
  embed.weight <- mx.symbol.Variable("embed.weight")
  rnn.params.weight <- mx.symbol.Variable("rnn.params.weight")
  rnn.state.weight <- mx.symbol.Variable("rnn.state.weight")
  if (cell.type == "lstm") rnn.state.cell.weight <- mx.symbol.Variable("rnn.state.cell.weight")
  cls.weight <- mx.symbol.Variable("cls.weight")
  cls.bias <- mx.symbol.Variable("cls.bias")
  
  data <- mx.symbol.transpose(data=data)  
  # seq.mask <- mx.symbol.stop_gradient(seq.mask, name="seq.mask")
  
  embed <- mx.symbol.Embedding(data=data, input_dim=input.size,
                               weight=embed.weight, output_dim=num.embed, name="embed")
  
  if (cell.type == "lstm") {
    rnn <- mx.symbol.RNN(data=embed, state=rnn.state.weight, state_cell = rnn.state.cell.weight, parameters=rnn.params.weight, state.size=num.hidden, num.layers=num.rnn.layer, bidirectional=F, mode=cell.type, state.outputs=F, p=dropout, name=paste(cell.type, num.rnn.layer, "layer", sep="_"))
    
  } else {
    rnn <- mx.symbol.RNN(data=embed, state=rnn.state.weight, parameters=rnn.params.weight, state.size=num.hidden, num.layers=num.rnn.layer, bidirectional=F, mode=cell.type, state.outputs=F, p=dropout, name=paste(cell.type, num.rnn.layer, "layer", sep="_"))
  }
  
  if (config=="seq-to-one") {
    
    if (masking) mask <- mx.symbol.SequenceLast(data=rnn[[1]], use.sequence.length = T, sequence_length = seq.mask, name = "mask") else
      mask <- mx.symbol.identity(data = rnn[[1]], name = "mask")
    
    fc <- mx.symbol.FullyConnected(data=mask,
                                   weight=cls.weight,
                                   bias=cls.bias,
                                   num.hidden=num.label,
                                   name = "decode")
    
    loss <- mx.symbol.SoftmaxOutput(data=fc, name="sm", label=label, ignore_label=ignore_label)
    
  } else if (config=="one-to-one"){
    
    if (masking) mask <- mx.symbol.SequenceMask(data = rnn[[1]], use.sequence.length = T, sequence_length = seq.mask, name = "mask") else
      mask <- mx.symbol.identity(data = rnn[[1]], name = "mask")
    
    reshape = mx.symbol.reshape(mask, shape=c(num.hidden, -1))
    
    fc <- mx.symbol.FullyConnected(data=reshape,
                                   weight=cls.weight,
                                   bias=cls.bias,
                                   num.hidden=num.label,
                                   name = "decode")
    
    label <- mx.symbol.reshape(data=label, shape=c(-1))
    loss <- mx.symbol.SoftmaxOutput(data=fc, name="sm", label=label, ignore_label=ignore_label)
    
  }
  
  if (output_last_state){
    # group <- mx.symbol.Group(c(unlist(last.states), loss))
    # return(group)
    return(loss)
  } else return(loss)
}



# data <- mx.symbol.Variable("data")
# reshape <- mx.symbol.reshape(data, shape=c(10, -1))
# fc <- mx.symbol.FullyConnected(reshape, num.hidden = 2)
# loss <- mx.symbol.SoftmaxOutput(fc)
# 
# graph.viz(loss, shape=c(10, 12, 64))
# 
# loss$infer.shape(list(data=c(10,12,64)))

# RNN test
# data <- mx.symbol.Variable("data")
# embed.weight <- mx.symbol.Variable("embed.weight")
# rnn.state.weight <- mx.symbol.Variable("rnn.state.weight")
# rnn.params.weight <- mx.symbol.Variable("rnn.params.weight")
# 
# batch.size <- 32
# seq.len <- 5
# input.size <- 50
# num.embed <- 8
# num.hidden <- 10
# seqidx <- 1
# num.rnn.layer <- 2
# 
# data <- mx.symbol.transpose(data=data)
# embed <- mx.symbol.Embedding(data=data, input_dim=input.size, weight=embed.weight, output_dim=num.embed, name="embed")
# rnn <- mx.symbol.RNN(data=embed, state=rnn.state.weight, parameters=rnn.params.weight, state.size=num.hidden, num.layers=num.rnn.layer, bidirectional=F, mode=cell.type, state.outputs=T, p=dropout, name=paste(cell.type, num.rnn.layer, "layer", seqidx, sep="_"))
# last.state <- mx.symbol.SequenceLast(rnn[[1]])
# 
# rnn$infer.shape(list(data=c(5, 32)))
# rnn$infer.shape(list(data=c(32, 5)))
# 
# last.state$infer.shape(list(data=c(5, 32)))
# last.state$infer.shape(list(data=c(32, 5)))
# 
# embed$infer.shape(list(data=c(5, 32)))
# embed$infer.shape(list(data=c(32, 5)))
# 
# embed <- mx.symbol.Embedding(data=data, input_dim=input.size, weight=embed.weight, output_dim=num.embed, name="embed")
# wordvec <- mx.symbol.split(data=embed, axis=1, num.outputs=seq.len, squeeze_axis=F)
# rnn <- mx.symbol.RNN(data=wordvec[[1]], state=rnn.state.weight, parameters=rnn.params.weight, state.size=num.hidden, num.layers=num.rnn.layer, bidirectional=F, mode=cell.type, state.outputs=T, p=dropout, name=paste(cell.type, num.rnn.layer, "layer", seqidx, sep="_"))
# last.state <- mx.symbol.SequenceLast(rnn[[1]])
# rnn$infer.shape(list(data=c(5, 32)))
