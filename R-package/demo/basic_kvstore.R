require(mxnet)

kv = mx.kv.create()

dlist = lapply(1:3, function(i) {
  x = as.array(c(i, i+1))
  mat = mx.nd.array(x, mx.cpu(i))
  list(x=mat)
})
kv$init(c(0), dlist[[1]])
kv$push(c(0), dlist, 0)
kv$pull(c(0), dlist, 0)

print(as.array(dlist[[1]][[1]]))




