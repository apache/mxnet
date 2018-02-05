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
  kv
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
  @test maximum(abs.(copy(val) .- 1)) == 0
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
    @test maximum(abs.(copy(v)) - num_devs) == 0
  end

  # list
  vals = [mx.NDArray[mx.ones(SHAPE, dev)*2 for dev in devs] for k in KEYS]
  mx.push!(kv, KEYS, vals)
  mx.pull!(kv, KEYS, vals)

  for vv in vals
    for v in vv
      @test maximum(abs.(copy(v)) - 2 * num_devs) == 0
    end
  end
end

function check_setupdater!(f)
  kv = KVStore(:local)
  setupdater!(kv, f)

  A = Float32[1, 2, 3, 4]
  B = Float32[.5, .6, .7, .8]
  x = NDArray(A)
  Δ = NDArray(B)
  init!(kv, 42, x)
  push!(kv, 42, Δ)
  pull!(kv, 42, x)

  @test copy(x) ≈ A + 2B
end  # function check_setupdater!

function test_setupdater!()
  info("KVStore::setupdater!")

  f(key, Δ, x) = @mx.inplace x += 2Δ
  g(key, Δ, x) = (x[:] += 2Δ)

  check_setupdater!(f)
  check_setupdater!(g)
end  # test_setupdater!

@testset "KVStore Test" begin
  test_kv_basic()
  test_single_kv_pair()
  test_aggregator()
  test_setupdater!()
end

end
