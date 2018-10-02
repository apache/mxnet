module TestRandom
using MXNet
using Base.Test

function test_uniform()
  dims = (100, 100, 2)
  info("random::uniform::dims = $dims")

  low = -10; high = 10
  seed = 123
  mx.srand(seed)
  ret1 = mx.rand(dims..., low = low, high = high)

  mx.srand(seed)
  ret2 = mx.empty(dims)
  mx.rand!(ret2, low = low, high = high)

  @test copy(ret1) == copy(ret2)
  @test abs(mean(copy(ret1)) - (high+low)/2) < 0.1
end

function test_gaussian()
  dims = (80, 80, 4)
  info("random::gaussian::dims = $dims")

  μ = 10; σ = 2
  seed = 456
  mx.srand(seed)
  ret1 = mx.randn(dims..., μ = μ, σ = σ)

  mx.srand(seed)
  ret2 = mx.empty(dims)
  mx.randn!(ret2, μ = μ, σ = σ)

  @test copy(ret1) == copy(ret2)
  @test abs(mean(copy(ret1)) - μ) < 0.1
  @test abs(std(copy(ret1)) - σ) < 0.1
end

@testset "Random Test" begin
  test_uniform()
  test_gaussian()
end

end
