module TestOptimizer

using Base.Test

using MXNet
using MXNet.mx.LearningRate
using MXNet.mx.Momentum


function test_fixed_η()
  info("Optimizer::LearningRate::Fixed")
  x = LearningRate.Fixed(.42)
  @test get(x) == .42
  update!(x)
  @test get(x) == .42
end  # function test_fixed_η


function check_η_decay(x)
  info("Optimizer::LearningRate::$x")

  η = get(x)
  @test η == 1

  for i ∈ 1:5
    update!(x)
    η′ = get(x)
    @test η′ < η
    η = η′
  end
end  # function check_η_decay


test_exp_η() = LearningRate.Exp(1) |> check_η_decay


test_inv_η() = LearningRate.Inv(1) |> check_η_decay


function test_μ_null()
  info("Optimizer::Momentum::Null")
  x = Momentum.Null()
  @test iszero(get(x))
end


function test_μ_fixed()
  info("Optimizer::Momentum::Fixed")
  x = Momentum.Fixed(42)
  @test get(x) == 42
end


@testset "Optimizer Test" begin
    @testset "LearningRate Test" begin
      test_fixed_η()
      test_exp_η()
      test_inv_η()
    end

    @testset "Momentum Test" begin
      test_μ_null()
      test_μ_fixed()
    end
end


end  # module TestOptimizer
