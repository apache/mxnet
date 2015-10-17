module TestNDArray
using MXNet
using Base.Test

using ..Main: rand_dims, reldiff

################################################################################
# Test Implementations
################################################################################
function rand_tensors{N}(dims::NTuple{N, Int})
  tensor = rand(mx.MX_float, dims)
  array  = copy(tensor, mx.DEFAULT_CONTEXT)
  return (tensor, array)
end

function test_copy()
  dims    = rand_dims()
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

function test_assign()
  dims    = rand_dims()
  tensor  = rand(mx.MX_float, dims)

  info("NDArray::assign::dims = $dims")

  # Julia Array -> NDArray assignment
  array   = mx.empty(size(tensor))
  array[:]= tensor
  @test reldiff(tensor, copy(array)) < 1e-6

  array2  = mx.zeros(size(tensor))
  @test reldiff(zeros(size(tensor)), copy(array2)) < 1e-6

  # scalar -> NDArray assignment
  scalar    = rand()
  array2[:] = scalar
  @test reldiff(zeros(size(tensor))+scalar, copy(array2)) < 1e-6

  # NDArray -> NDArray assignment
  array[:]  = array2
  @test reldiff(zeros(size(tensor))+scalar, copy(array)) < 1e-6
end

function test_slice()
  array = mx.zeros((2,4))
  array[2:3] = ones(2,2)
  @test copy(array) == [0 1 1 0; 0 1 1 0]
  @test copy(slice(array, 2:3)) == [1 1; 1 1]
end

function test_plus()
  dims   = rand_dims()
  t1, a1 = rand_tensors(dims)
  t2, a2 = rand_tensors(dims)
  t3, a3 = rand_tensors(dims)

  info("NDArray::plus::dims = $dims")

  @test reldiff(t1+t2, copy(a1+a2)) < 1e-6
  @test reldiff(t1.+t2, copy(a1.+a2)) < 1e-6

  @test reldiff(t1+t2+t3, copy(a1+a2+a3)) < 1e-6

  # test inplace += operation
  a0 = a1               # keep a reference to a1
  @mx.inplace a1 += a2  # perform inplace +=
  @test a0 == a1        # make sure they are still the same object
  @test reldiff(copy(a0), copy(a1)) < 1e-6
  @test reldiff(copy(a1), t1+t2) < 1e-6

  # test scalar
  scalar = rand()
  @test reldiff(t3 + scalar, copy(a3 + scalar)) < 1e-6
  @test reldiff(t2+scalar+t3, copy(a2+scalar+a3)) < 1e-6
end

function test_minus()
  dims   = rand_dims()
  t1, a1 = rand_tensors(dims)
  t2, a2 = rand_tensors(dims)

  info("NDArray::minus::dims = $dims")

  @test reldiff(t1-t2, copy(a1-a2)) < 1e-6
  @test reldiff(t1.-t2, copy(a1.-a2)) < 1e-6

  @test reldiff(-t1, copy(-a1)) < 1e-6

  # make sure the negation is not in-place, so a1 is not changed after previous
  # statement is executed
  @test reldiff(t1, copy(a1)) < 1e-6

  # test inplace -= operation
  a0 = a1              # keep a reference to a1
  @mx.inplace a1 -= a2 # perform inplace -=
  @test a0 == a1       # make sure they are still the same object
  @test reldiff(copy(a0), copy(a1)) < 1e-6
  @test reldiff(copy(a1), t1-t2) < 1e-6

  # test scalar
  scalar = rand()
  @test reldiff(t2 - scalar, copy(a2 - scalar)) < 1e-6
end

function test_mul()
  dims   = rand_dims()
  t1, a1 = rand_tensors(dims)
  t2, a2 = rand_tensors(dims)
  t3, a3 = rand_tensors(dims)

  info("NDArray::mul::dims = $dims")

  @test reldiff(t1.*t2, copy(a1.*a2)) < 1e-6

  # test inplace .*= operation
  a0 = a1               # keep a reference to a1
  @mx.inplace a1 .*= a2 # perform inplace .*=
  @test a0 == a1        # make sure they are still the same object
  @test reldiff(copy(a0), copy(a1)) < 1e-6
  @test reldiff(copy(a1), t1.*t2) < 1e-6

  # test scalar
  scalar = rand()
  @test reldiff(t3 * scalar, copy(a3 .* scalar)) < 1e-6
end

function test_div()
  dims   = rand_dims()
  t1, a1 = rand_tensors(dims)
  t2, a2 = rand_tensors(dims)

  info("NDArray::div::dims = $dims")
  t2             .+= 2  # avoid numerical instability
  @mx.inplace a2 .+= 2

  @test reldiff(t1 ./ t2, copy(a1 ./ a2)) < 1e-6

  # test inplace -= operation
  a0 = a1                # keep a reference to a2
  @mx.inplace a1 ./= a2  # perform inplace ./=
  @test a0 == a1         # make sure they are still the same object
  @test reldiff(copy(a0), copy(a1)) < 1e-6
  @test reldiff(copy(a1), t1 ./ t2) < 1e-6

  # test scalar
  scalar = rand() + 2
  @test reldiff(t2./scalar, copy(a2./scalar)) < 1e-6
end

function test_gd()
  dims   = rand_dims()
  tw, aw = rand_tensors(dims)
  tg, ag = rand_tensors(dims)

  info("NDArray::gd::dims = $dims")

  lr = rand()
  wd = rand()

  @mx.inplace aw += -lr * (ag + wd * aw)
  tw += -lr * (tg + wd * tw)
  @test reldiff(copy(aw), tw) < 1e-6
end


################################################################################
# Run tests
################################################################################
test_copy()
test_assign()
test_slice()
test_plus()
test_minus()
test_mul()
test_div()
test_gd()

end
