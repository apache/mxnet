module TestUtil

using Base.Test

using MXNet


function test_firstarg()
  info("Util::_firstarg")
  @test mx._firstarg(:(f(x, y))) == :x
  @test mx._firstarg(:(f(x::mx.NDArray, y))) == :x
  @test mx._firstarg(:(f(x::mx.NDArray, y::mx.NDArray))) == :x
  @test mx._firstarg(:(f(x::Int, y::mx.NDArray))) == :x
  @test mx._firstarg(:(f(x::Int, y::mx.NDArray; other = 42))) == :x
  @test mx._firstarg(:(f(x::mx.NDArray{T}, y) where {T})) == :x
  @test mx._firstarg(:(f(x::mx.NDArray{T,N}, y) where {T,N})) == :x
  @test mx._firstarg(:(f(x::mx.NDArray{T,N} where {T,N}, y))) == :x
  @test mx._firstarg(:(broadcast_(::typeof(asin), x::mx.NDArray))) == :x
  @test mx._firstarg(:(broadcast_(::typeof(asin), x::mx.NDArray, y::mx.NDArray))) == :x
end  # function test_firstarg


@testset "Util Test" begin
  test_firstarg()
end  # @testset "Util"

end  # module TestUtil
