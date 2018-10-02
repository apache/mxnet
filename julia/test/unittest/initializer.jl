@testset "Initializers" begin
  @testset "Bilinear initializer" begin
    # Setup a filter with scale = 2
    expectedFilter = Float32[
                       0.0625 0.1875 0.1875 0.0625;
                       0.1875 0.5625 0.5625 0.1875;
                       0.1875 0.5625 0.5625 0.1875;
                       0.0625 0.1875 0.1875 0.0625]
    filter = mx.zeros(Float32, 4, 4, 1, 4)
    mx.init(mx.XavierInitializer(), :upsampling0_weight, filter)

    mx.@nd_as_jl ro=filter begin
      for s in 1:size(filter, 4)
        @test all(filter[:, :, 1, s] .== expectedFilter)
      end
    end
  end
end
