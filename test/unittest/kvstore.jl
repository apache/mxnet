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

function test_kv_basic()
  info("KVStore::basic")

  kv = init_kv()
  @test mx.get_type(kv) == :local
  @test mx.get_rank(kv) == 0
  @test mx.get_num_workers(kv) == 1
end

function test_single_kv_pair()
  info("KVStore::single")

  kv = init_kv()
  mx.push!(kv, 3, mx.ones(SHAPE))
  val = mx.empty(SHAPE)
  mx.pull!(kv, 3, val)
  @test maximum(abs(copy(val) - 1)) == 0
end

function test_aggregator()
  info("KVStore::aggregator")

  kv = init_kv()

  num_devs = 4
  devs = [mx.Context(mx.CPU, i) for i=0:num_devs-1]
  vals = [mx.ones(SHAPE, dev) for dev in devs]

  mx.push!(kv, 3, vals)
  mx.pull!(kv, 3, vals)
  for v in vals
    @test maximum(abs(copy(v)) - num_devs) == 0
  end

  # list
  vals = [mx.NDArray[mx.ones(SHAPE, dev)*2 for dev in devs] for k in KEYS]
  mx.push!(kv, KEYS, vals)
  mx.pull!(kv, KEYS, vals)

  for vv in vals
    for v in vv
      @test maximum(abs(copy(v)) - 2num_devs) == 0
    end
  end
end

test_kv_basic()
test_single_kv_pair()
test_aggregator()

end
