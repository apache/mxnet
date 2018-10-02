module TestModel

using Base.Test
using MXNet


function test_feedforward()
  info("Model::FeedForward::constructor")
  let x = @mx.var x
    m = mx.FeedForward(x)
    @test m.arch === x
    @test length(m.ctx) == 1
  end

  info("Model::FeedForward::constructor::keyword context")
  let x = @mx.var x
    m = mx.FeedForward(x, context = mx.cpu())
    @test m.arch === x
    @test length(m.ctx) == 1
  end

  let x = @mx.var x
    m = mx.FeedForward(x, context = [mx.cpu(), mx.cpu(1)])
    @test m.arch === x
    @test length(m.ctx) == 2
  end
end


@testset "Model Test" begin
  test_feedforward()
end

end  # module TestModel
