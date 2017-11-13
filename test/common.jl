################################################################################
# Common models used in testing
################################################################################
function rand_dims(max_ndim=6)
  tuple(rand(1:10, rand(1:max_ndim))...)
end

function mlp2()
  data = mx.Variable(:data)
  out = mx.FullyConnected(data, name=:fc1, num_hidden=1000)
  out = mx.Activation(out, act_type=:relu)
  out = mx.FullyConnected(out, name=:fc2, num_hidden=10)
  return out
end

function mlpchain()
  mx.@chain mx.Variable(:data) =>
            mx.FullyConnected(name=:fc1, num_hidden=1000) =>
            mx.Activation(act_type=:relu) =>
            mx.FullyConnected(name=:fc2, num_hidden=10)
end

"""
execution helper of SymbolicNode
"""
function exec(x::mx.SymbolicNode; feed...)
  ks, vs = zip(feed...)
  vs′ = mx.NDArray.(vs)

  e = mx.bind(x, context = mx.cpu(), args = Dict(zip(ks, vs′)))
  mx.forward(e)
  e.outputs
end
