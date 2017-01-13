module TestNDArray
using MXNet
if VERSION ≥ v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

using ..Main: rand_dims, reldiff

################################################################################
# Test Implementations
################################################################################
rand_tensors{N}(dims::NTuple{N, Int}) = rand_tensors(mx.MX_float, dims)
function rand_tensors{N, T}(::Type{T}, dims::NTuple{N, Int})
  tensor = rand(T, dims)
  array  = copy(tensor, mx.cpu())
  return (tensor, array)
end

function test_copy()
  dims    = rand_dims()
  tensor  = rand(mx.MX_float, dims)

  info("NDArray::copy::dims = $dims")

  # copy to NDArray and back
  array   = copy(tensor, mx.cpu())
  tensor2 = copy(array)
  @test reldiff(tensor, tensor2) < 1e-6

  # copy between NDArray
  array2  = copy(array, mx.cpu())
  tensor2 = copy(array2)
  @test reldiff(tensor, tensor2) < 1e-6
end

function test_assign()
  dims    = rand_dims()
  tensor  = rand(mx.MX_float, dims)
  thresh  = 1e-3

  info("NDArray::assign::dims = $dims")

  # Julia Array -> NDArray assignment
  array   = mx.empty(size(tensor))
  array[:]= tensor
  @test reldiff(tensor, copy(array)) < thresh

  array2  = mx.zeros(size(tensor))
  @test reldiff(zeros(size(tensor)), copy(array2)) < thresh

  array3 = mx.zeros(Float16, size(tensor))
  @test reldiff(zeros(Float16, size(tensor)), copy(array2)) < thresh

  # scalar -> NDArray assignment
  scalar    = rand()
  array2[:] = scalar
  @test reldiff(zeros(size(tensor))+scalar, copy(array2)) < thresh

  scalar = rand(Float16)
  array2[:] = scalar
  @test reldiff(zeros(size(tensor))+scalar, copy(array2)) < thresh

  scalar = rand(Float64)
  array2[:] = scalar
  array3[:] = scalar
  @test reldiff(zeros(size(tensor))+scalar, copy(array2)) < thresh
  @test reldiff(zeros(Float16,size(tensor))+scalar, copy(array3)) < thresh

  # NDArray -> NDArray assignment
  array[:]  = array2
  @test reldiff(zeros(size(tensor))+scalar, copy(array)) < thresh
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
  thresh = 1e-6

  info("NDArray::plus::dims = $dims")

  @test reldiff(t1+t2, copy(a1+a2)) < thresh
  @test reldiff(t1.+t2, copy(a1.+a2)) < thresh

  @test reldiff(t1+t2+t3, copy(a1+a2+a3)) < thresh

  # test inplace += operation
  a0 = a1               # keep a reference to a1
  @mx.inplace a1 += a2  # perform inplace +=
  @test a0 == a1        # make sure they are still the same object
  @test reldiff(copy(a0), copy(a1)) < thresh
  @test reldiff(copy(a1), t1+t2) < thresh

  # test scalar
  scalar = rand()
  @test reldiff(t3 + scalar, copy(a3 + scalar)) < thresh
  @test reldiff(t2+scalar+t3, copy(a2+scalar+a3)) < thresh

  # test small and large scalar
  t4 = zeros(Float32, dims)
  a4 = copy(t4, mx.cpu())
  scalar_small = 1e-8
  scalar_large = 1e8
  @test reldiff(t4 + scalar_small, copy(a4 .+ scalar_small)) < thresh
  @test reldiff(t4 + scalar_large, copy(a4 .+ scalar_large)) < thresh
  
  t5 = zeros(Float64, dims)
  a5 = copy(t5, mx.cpu())
  scalar_small = 1e-8
  scalar_large = 1e8
  @test reldiff(t5 + scalar_small, copy(a5 .+ scalar_small)) < thresh
  @test reldiff(t5 + scalar_large, copy(a5 .+ scalar_large)) < thresh

  t6 = zeros(Float16, dims)
  a6 = copy(t6, mx.cpu())
  scalar_small = Float16(1e-5)
  scalar_large = Float16(1e4)
  @test reldiff(t6 + scalar_small, copy(a6 .+ scalar_small)) < 1e-1
  @test reldiff(t6 + scalar_large, copy(a6 .+ scalar_large)) < 1e-1
end

function test_minus()
  dims   = rand_dims()
  t1, a1 = rand_tensors(dims)
  t2, a2 = rand_tensors(dims)
  thresh = 1e-6

  info("NDArray::minus::dims = $dims")

  @test reldiff(t1-t2, copy(a1-a2)) < thresh
  @test reldiff(t1.-t2, copy(a1.-a2)) < thresh

  @test reldiff(-t1, copy(-a1)) < thresh

  # make sure the negation is not in-place, so a1 is not changed after previous
  # statement is executed
  @test reldiff(t1, copy(a1)) < thresh

  # test inplace -= operation
  a0 = a1              # keep a reference to a1
  @mx.inplace a1 -= a2 # perform inplace -=
  @test a0 == a1       # make sure they are still the same object
  @test reldiff(copy(a0), copy(a1)) < thresh
  @test reldiff(copy(a1), t1-t2) < thresh

  # test scalar
  scalar = rand()
  @test reldiff(t2 - scalar, copy(a2 - scalar)) < thresh

  # test small and large scalar
  t4 = zeros(Float32, dims)
  a4 = copy(t4, mx.cpu())
  scalar_small = 1e-8
  scalar_large = 1e8
  @test reldiff(t4 - scalar_small, copy(a4 .- scalar_small)) < thresh
  @test reldiff(t4 - scalar_large, copy(a4 .- scalar_large)) < thresh
  
  t5 = zeros(Float64, dims)
  a5 = copy(t5, mx.cpu())
  scalar_small = 1e-8
  scalar_large = 1e8
  @test reldiff(t5 - scalar_small, copy(a5 .- scalar_small)) < thresh
  @test reldiff(t5 - scalar_large, copy(a5 .- scalar_large)) < thresh

  t6 = zeros(Float16, dims)
  a6 = copy(t6, mx.cpu())
  scalar_small = Float16(1e-5)
  scalar_large = Float16(1e4)
  @test reldiff(t6 - scalar_small, copy(a6 .- scalar_small)) < 1e-1
  @test reldiff(t6 - scalar_large, copy(a6 .- scalar_large)) < 1e-1
end

function test_mul()
  dims   = rand_dims()
  t1, a1 = rand_tensors(dims)
  t2, a2 = rand_tensors(dims)
  t3, a3 = rand_tensors(dims)
  thresh = 1e-6

  info("NDArray::mul::dims = $dims")

  @test reldiff(t1.*t2, copy(a1.*a2)) < thresh

  # test inplace .*= operation
  a0 = a1               # keep a reference to a1
  @mx.inplace a1 .*= a2 # perform inplace .*=
  @test a0 == a1        # make sure they are still the same object
  @test reldiff(copy(a0), copy(a1)) < thresh
  @test reldiff(copy(a1), t1.*t2) < thresh

  # test scalar
  scalar = mx.MX_float(rand())
  @test reldiff(t3 * scalar, copy(a3 .* scalar)) < thresh

  # test small and large scalar
  t4, a4 = rand_tensors(Float32, dims)
  scalar_small = 1e-8
  scalar_large = 1e8
  @test reldiff(t4 * scalar_small, copy(a4 .* scalar_small)) < thresh
  @test reldiff(t4 * scalar_large, copy(a4 .* scalar_large)) < thresh
  
  t5, a5 = rand_tensors(Float64, dims)
  scalar_small = 1e-8
  scalar_large = 1e8
  @test reldiff(t5 * scalar_small, copy(a5 .* scalar_small)) < thresh
  @test reldiff(t5 * scalar_large, copy(a5 .* scalar_large)) < thresh

  t6, a6 = rand_tensors(Float16, dims)
  scalar_small = Float16(1e-5)
  @test reldiff(t6 * scalar_small, copy(a6 .* scalar_small)) < 1e-1
end

function test_div()
  dims   = rand_dims()
  t1, a1 = rand_tensors(dims)
  t2, a2 = rand_tensors(dims)
  thresh = 1e-6

  info("NDArray::div::dims = $dims")
  t2             .+= 2  # avoid numerical instability
  @mx.inplace a2 .+= 2

  @test reldiff(t1 ./ t2, copy(a1 ./ a2)) < thresh

  # test inplace -= operation
  a0 = a1                # keep a reference to a2
  @mx.inplace a1 ./= a2  # perform inplace ./=
  @test a0 == a1         # make sure they are still the same object
  @test reldiff(copy(a0), copy(a1)) < thresh
  @test reldiff(copy(a1), t1 ./ t2) < thresh

  # test scalar
  scalar = rand() + 2
  @test reldiff(t2./scalar, copy(a2./scalar)) < thresh

  # test small and large scalar
  t4, a4 = rand_tensors(Float32, dims)
  scalar_small = 1e-8
  scalar_large = 1e8
  @test reldiff(t4 / scalar_small, copy(a4 ./ scalar_small)) < thresh
  @test reldiff(t4 / scalar_large, copy(a4 ./ scalar_large)) < thresh
  
  t5, a5 = rand_tensors(Float64, dims)
  scalar_small = 1e-8
  scalar_large = 1e8
  @test reldiff(t5 / scalar_small, copy(a5 ./ scalar_small)) < thresh
  @test reldiff(t5 / scalar_large, copy(a5 ./ scalar_large)) < thresh

  t6, a6 = rand_tensors(Float16, dims)
  scalar_large = 1e4
  @test reldiff(t6 / scalar_large, copy(a6 ./ scalar_large)) < 1e-1
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


function test_saveload()
  n_arrays = 5
  info("NDArray::saveload::n_arrays = $n_arrays")
  fname = tempname()

  # save and load a single array
  dims   = rand_dims()
  j_array, nd_array = rand_tensors(dims)
  mx.save(fname, nd_array)
  data = mx.load(fname, mx.NDArray)
  @test isa(data, Vector{mx.NDArray})
  @test length(data) == 1
  @test reldiff(copy(data[1]), j_array) < 1e-6

  # save and load N arrays of different shape
  arrays = [rand_tensors(rand_dims()) for i = 1:n_arrays]
  nd_arrays = mx.NDArray[x[2] for x in arrays]
  mx.save(fname, nd_arrays)
  data = mx.load(fname, mx.NDArray)
  @test isa(data, Vector{mx.NDArray})
  @test length(data) == n_arrays
  for i = 1:n_arrays
    @test reldiff(copy(data[i]), arrays[i][1]) < 1e-6
  end

  # save and load dictionary of ndarrays
  names = [Symbol("array$i") for i = 1:n_arrays]
  dict = Dict([(n, v) for (n,v) in zip(names, nd_arrays)])
  mx.save(fname, dict)
  data = mx.load(fname, mx.NDArray)
  @test isa(data, Dict{Symbol, mx.NDArray})
  @test length(data) == n_arrays
  for i = 1:n_arrays
    @test reldiff(copy(data[names[i]]), arrays[i][1]) < 1e-6
  end

  rm(fname)
end

function test_clip()
  dims = rand_dims()
  info("NDArray::clip::dims = $dims")

  j_array, nd_array = rand_tensors(dims)
  clip_up   = maximum(abs(j_array)) / 2
  clip_down = 0
  clipped   = mx.clip(nd_array, a_min=clip_down, a_max=clip_up)

  # make sure the original array is not modified
  @test reldiff(copy(nd_array), j_array) < 1e-6

  @test all(clip_down .<= copy(clipped) .<= clip_up)
end

function test_sqrt()
  dims = rand_dims()
  info("NDArray::sqrt::dims = $dims")

  j_array, nd_array = rand_tensors(dims)
  sqrt_ed = sqrt(nd_array)
  @test reldiff(copy(sqrt_ed), sqrt(j_array)) < 1e-6
end

function test_nd_as_jl()
  dims = (2,3)
  info("NDArray::nd_as_jl::dims = $dims")

  x = mx.zeros(dims) + 5
  y = mx.ones(dims)
  z = mx.zeros(dims)
  @mx.nd_as_jl ro=x rw=(y,z) begin
    for i = 1:length(z)
      z[i] = x[i]
    end

    z[:,1] = y[:,1]
    y[:] = 0
  end

  @test reldiff(copy(y), 0) < 1e-6
  @test reldiff(copy(z)[:,1], 1) < 1e-6
  @test reldiff(copy(z)[:,2:end], copy(x)[:,2:end]) < 1e-6
end

function test_dot()
  dims1 = (2, 3)
  dims2 = (3, 8)
  info("NDArray::dot")

  x = mx.zeros(dims1)
  y = mx.zeros(dims2)
  z = mx.dot(x, y)
  @test size(z) == (2, 8)
end

function test_eltype()
  info("NDArray::eltype")
  dims1 = (3,3)

  x = mx.empty(dims1)
  @test eltype(x) == mx.DEFAULT_DTYPE

  for TF in instances(mx.TypeFlag)
    T = mx.fromTypeFlag(TF)
    x = mx.empty(T, dims1)
    @test eltype(x) == T
  end
end

function test_kwargs()
  info("NDArray::kwargs")
  dims1 = (2,3,4)

  A = rand(Float32, dims1)
  x = mx.NDArray(A)
  tx = mx.transpose(x, axes=(2,1,3))
  tA = permutedims(A, [2,1,3])
  @test size(tx) == size(tA)
  @test all(copy(tx) .== tA)
end

################################################################################
# Run tests
################################################################################
@testset "NDArray Test" begin
  test_assign()
  test_copy()
  test_slice()
  test_plus()
  test_minus()
  test_mul()
  test_div()
  test_gd()
  test_saveload()
  test_clip()
  test_sqrt()
  test_eltype()
  test_nd_as_jl()
  test_dot()
  test_kwargs()
end

end
