################################################################################
# Common models used in testing
################################################################################
function mlp2()
  data = mx.variable(:data)
  out = mx.FullyConnected(data=data, name=:fc1, num_hidden=1000)
  out = mx.Activation(data=out, act_type=:relu)
  out = mx.FullyConnected(data=out, name=:fc2, num_hidden=10)
  return out
end
