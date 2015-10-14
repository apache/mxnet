module TestKVStore
using MXNet
using Base.Test

using ..Main: rand_dims

SHAPE = rand_dims()
KEYS  = [5,7,11]

function init_kv()
  kv = mx.KVStore()
  mx.init!(kv, 3, mx.zeros(SHAPE))

  vals = [mx.zeros(SHAPE) for k in KEYS]
  mx.init!(kv, KEYS, vals)
  return kv
end

function test_single_kv_pair()
  info("KVStore::single")

  kv = init_kv()
  mx.push!(kv, 3, mx.ones(SHAPE))
  val = mx.empty(SHAPE)
  mx.pull!(kv, 3, val)
  @test maximum(abs(copy(val) - 1)) == 0
end

test_single_kv_pair()

end
