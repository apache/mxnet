module TestNDArray
using MXNet
using Base.Test

################################################################################
# Test Implementations
################################################################################
function reldiff(a, b)
  diff = sum(abs(a - b))
  norm = sum(abs(a))
  return diff / norm
end

function test_copy()
  dims    = tuple(rand(1:10, rand(1:6))...)
  tensor  = rand(mx.MX_float, dims)

  info("NDArray::copy::dims = $dims")

  # copy to NDArray and back
  array   = copy(tensor, mx.DEFAULT_CONTEXT)
  tensor2 = copy(array)
  @test reldiff(tensor, tensor2) < 1e-6

  # copy between NDArray
  array2  = copy(array, mx.DEFAULT_CONTEXT)
  tensor2 = copy(array2)
  @test reldiff(tensor, tensor2) < 1e-6
end


################################################################################
# Run tests
################################################################################
test_copy()

end
