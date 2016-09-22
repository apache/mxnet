################################################################################
# Common models used in testing
################################################################################
function reldiff(a, b)
  diff = sum(abs(a - b))
  norm = sum(abs(a))
  return diff / (norm + 1e-10)
end

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

