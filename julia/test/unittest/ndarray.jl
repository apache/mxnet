# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

module TestNDArray

using MXNet
using Statistics
using LinearAlgebra
using Test

using ..Main: rand_dims

################################################################################
# Test Implementations
################################################################################
rand_tensors(dims::NTuple{N,Int}) where {N} = rand_tensors(mx.MX_float, dims)
function rand_tensors(::Type{T}, dims::NTuple{N,Int}) where {N,T}
  tensor = rand(T, dims)
  array  = copy(tensor, mx.cpu())
  return (tensor, array)
end

function test_constructor()
  @info("NDArray::NDArray(x::AbstractArray)")
  function check_absarray(x)
    y = mx.NDArray(x)
    @test ndims(x)  == ndims(y)
    @test eltype(x) == eltype(y)
    @test x[3]      == y[3][]
  end

  check_absarray(1:10)
  check_absarray(1.0:10)

  @info("NDArray::NDArray(Type, AbstractArray)")
  let
    x = mx.NDArray(Float32, [1, 2, 3])
    @test eltype(x) == Float32
    @test copy(x) == [1, 2, 3]
  end
  let
    x = mx.NDArray(Float32, [1.1, 2, 3])
    @test eltype(x) == Float32
    @test copy(x) ≈ [1.1, 2, 3]
  end

  @info "NDArray::NDArray{T,N}(undef, dims...)"
  let
    x = NDArray{Int,2}(undef, 5, 5)
    @test eltype(x) == Int
    @test size(x) == (5, 5)
    @test x.writable

    y = NDArray{Int,2}(undef, 5, 5, writable = false)
    @test !y.writable

    # dimension mismatch
    @test_throws MethodError NDArray{Int,1}(undef, 5, 5)
  end

  @info "NDArray::NDArray{T,N}(undef, dims)"
  let
    x = NDArray{Int,2}(undef, (5, 5))
    @test eltype(x) == Int
    @test size(x) == (5, 5)
    @test x.writable

    y = NDArray{Int,2}(undef, (5, 5), writable = false)
    @test !y.writable

    # dimension mismatch
    @test_throws MethodError NDArray{Int,1}(undef, (5, 5))
  end

  @info "NDArray::NDArray{T}(undef, dims...)"
  let
    x = NDArray{Int}(undef, 5, 5)
    @test eltype(x) == Int
    @test size(x) == (5, 5)
    @test x.writable

    y = NDArray{Int}(undef, 5, 5, writable = false)
    @test !y.writable
  end

  @info "NDArray::NDArray{T}(undef, dims)"
  let
    x = NDArray{Int}(undef, (5, 5))
    @test eltype(x) == Int
    @test size(x) == (5, 5)
    @test x.writable

    y = NDArray{Int}(undef, (5, 5), writable = false)
    @test !y.writable
  end

  @info "NDArray::NDArray(undef, dims...)"
  let
    x = NDArray(undef, 5, 5)
    @test eltype(x) == mx.MX_float
    @test size(x) == (5, 5)
    @test x.writable

    y = NDArray(undef, 5, 5, writable = false)
    @test !y.writable
  end

  @info "NDArray::NDArray(undef, dims)"
  let
    x = NDArray(undef, (5, 5))
    @test eltype(x) == mx.MX_float
    @test size(x) == (5, 5)
    @test x.writable

    y = NDArray(undef, (5, 5), writable = false)
    @test !y.writable
  end
end  # function test_constructor


function test_ones_zeros_like()
  @info("NDArray::Base.zeros")
  let x = mx.rand(1, 3, 2, 4, low = 1, high = 10)
    y = zeros(x)
    @test sum(copy(y)) == 0

    y = mx.zeros(x)
    @test sum(copy(y)) == 0
  end

  @info("NDArray::Base.ones")
  let x = mx.rand(1, 3, 2, 4, low = 1, high = 10)
    y = ones(x)
    @test sum(copy(y)) == 1 * 3 * 2 * 4

    y = mx.ones(x)
    @test sum(copy(y)) == 1 * 3 * 2 * 4
  end
end  # function test_ones_zeros_like


function test_copy()
  dims    = rand_dims()
  tensor  = rand(mx.MX_float, dims)

  @info("NDArray::copy::dims = $dims")

  # copy to NDArray and back
  array   = copy(tensor, mx.cpu())
  tensor2 = copy(array)
  @test tensor ≈ tensor2

  # copy between NDArray
  array2  = copy(array, mx.cpu())
  tensor2 = copy(array2)
  @test tensor ≈ tensor2

  @info("NDArray::copy::AbstractArray")
  let x = copy(1:4, mx.cpu())
    @test eltype(x) == Int
    @test copy(x) == [1, 2, 3, 4]
  end

  let x = copy(1.:4, mx.cpu())
    @test eltype(x) == Float64
    @test copy(x) ≈ [1., 2, 3, 4]
  end

  @info("NDArray::copy!::AbstractArray")
  let
    x = mx.zeros(4)
    copy!(x, 1:4)

    @test eltype(x) == Float32
    @test copy(x) == [1, 2, 3, 4]
  end
end

function test_deepcopy()
  @info("NDArray::deepcopy")

  x = mx.zeros(2, 5)
  y = deepcopy(x)
  x[:] = 42
  @test copy(x) != copy(y)
end

function test_assign()
  dims    = rand_dims()
  tensor  = rand(mx.MX_float, dims)

  @info("NDArray::assign::dims = $dims")

  # Julia Array -> NDArray assignment
  array    = NDArray(undef, size(tensor)...)
  array[:] = tensor
  @test tensor ≈ copy(array)

  array2  = mx.zeros(size(tensor))
  @test zeros(size(tensor)) ≈ copy(array2)

  array3 = mx.zeros(Float16, size(tensor))
  @test zeros(Float16, size(tensor)) ≈ copy(array2)

  # scalar -> NDArray assignment
  scalar    = rand()
  array2[:] = scalar
  @test zeros(size(tensor)) .+ scalar ≈ copy(array2)

  scalar = rand(Float16)
  array2[:] = scalar
  @test zeros(size(tensor)) .+ scalar ≈ copy(array2)

  scalar = rand(Float64)
  array2[:] = scalar
  array3[:] = scalar
  @test zeros(size(tensor)) .+ scalar ≈ copy(array2)
  @test zeros(Float16, size(tensor)) .+ scalar ≈ copy(array3)

  # NDArray -> NDArray assignment
  array[:]  = array2
  @test zeros(size(tensor)) .+ scalar ≈ copy(array)
end

function test_slice()
  array = mx.zeros((2, 4))
  array[2:3] = ones(2, 2)
  @test copy(array) == [0 1 1 0; 0 1 1 0]
  @test copy(mx.slice(array, 2:3)) == [1 1; 1 1]
end

function test_linear_idx()
  @info("NDArray::getindex::linear indexing")
  let A = reshape(1:30, 3, 10)
    x = mx.NDArray(A)

    @test copy(x) == A
    @test copy(x[1])  == [1]
    @test copy(x[2])  == [2]
    @test copy(x[3])  == [3]
    @test copy(x[12]) == [12]
    @test copy(x[13]) == [13]
    @test copy(x[14]) == [14]

    @test_throws BoundsError x[-1]
    @test_throws BoundsError x[0]
    @test_throws BoundsError x[31]
    @test_throws BoundsError x[42]
  end

  let A = reshape(1:24, 3, 2, 4)
    x = mx.NDArray(A)

    @test copy(x) == A
    @test copy(x[1])  == [1]
    @test copy(x[2])  == [2]
    @test copy(x[3])  == [3]
    @test copy(x[11]) == [11]
    @test copy(x[12]) == [12]
    @test copy(x[13]) == [13]
    @test copy(x[14]) == [14]
  end

  @info("NDArray::setindex!::linear indexing")
  let A = reshape(1:24, 3, 2, 4)
    x = mx.NDArray(A)

    @test copy(x) == A

    x[4] = -4
    @test copy(x[4])  == [-4]

    x[11] = -11
    @test copy(x[11]) == [-11]

    x[24] = 42
    @test copy(x[24]) == [42]
  end

  @info("NDArray::setindex!::type convert")
  let
    x = NDArray([1, 2, 3])
    @test eltype(x) == Int
    x[:] = π
    @test copy(x) == [3, 3, 3]
  end
end  # function test_linear_idx

function test_first()
  @info("NDArray::first")
  let A = reshape(1:30, 3, 10)
    x = mx.NDArray(A)

    @test x[]    == 1
    @test x[5][] == 5

    @test first(x)    == 1
    @test first(x[5]) == 5
  end
end  # function test_first

function test_lastindex()
  @info("NDArray::lastindex")
  let A = [1 2; 3 4; 5 6], x = mx.NDArray(A)
    @test lastindex(A) == lastindex(x)
  end
end  # function test_lastindex

function test_cat()
  function check_cat(f, A, B = 2A)
    C = [A B]
    D = [A; B]
    x = NDArray(A)
    y = NDArray(B)
    z = NDArray(C)
    d = NDArray(D)

    if f == :hcat
      @test copy([x y]) == [A B]
      @test copy([x y 3y x]) == [A B 3B A]
      @test copy([z y x]) == [C B A]
    elseif f == :vcat
      @test copy([x; y]) == [A; B]
      @test copy([x; y; 3y; x]) == [A; B; 3B; A]
      @test copy([x; d]) == [A; D]
      @test copy([d; x]) == [D; A]
    else
      @assert false
    end
  end

  let A = [1, 2, 3, 4]
    @info("NDArray::hcat::1D")
    check_cat(:hcat, A)

    @info("NDArray::vcat::1D")
    check_cat(:vcat, A)
  end

  let A = [1 2; 3 4]
    @info("NDArray::hcat::2D")
    check_cat(:hcat, A)

    @info("NDArray::vcat::2D")
    check_cat(:vcat, A)
  end

  let A = rand(4, 3, 2)
    @info("NDArray::hcat::3D")
    check_cat(:hcat, A)

    @info("NDArray::vcat::3D")
    check_cat(:vcat, A)
  end

  let A = rand(4, 3, 2, 2)
    @info("NDArray::hcat::4D")
    check_cat(:hcat, A)

    @info("NDArray::vcat::4D")
    check_cat(:vcat, A)
  end

  let A = [1, 2, 3, 4]
    @info("NDArray::cat::3D/1D")
    check_cat(:vcat, reshape(A, 4, 1, 1), 2A)
  end
end  # function test_cat

function test_plus()
  dims   = rand_dims()
  t1, a1 = rand_tensors(dims)
  t2, a2 = rand_tensors(dims)
  t3, a3 = rand_tensors(dims)

  @info("NDArray::plus::dims = $dims")

  @test t1 .+ t2 ≈ copy(a1 .+ a2)

  @test t1 .+ t2 .+ t3 ≈ copy(a1 .+ a2 .+ a3)

  # test inplace += operation
  a0 = a1               # keep a reference to a1
  @mx.inplace a1 += a2  # perform inplace +=
  @test a0 == a1        # make sure they are still the same object
  @test copy(a0) ≈ copy(a1)
  @test copy(a1) ≈ t1 .+ t2

  # test scalar
  scalar = rand()
  @test t3 .+ scalar       ≈ copy(a3 .+ scalar)
  @test t2 .+ scalar .+ t3 ≈ copy(a2 .+ scalar .+ a3)

  # test small and large scalar
  t4 = zeros(Float32, dims)
  a4 = copy(t4, mx.cpu())
  scalar_small = 1e-8
  scalar_large = 1e8
  @test t4 .+ scalar_small ≈ copy(a4 .+ scalar_small)
  @test t4 .+ scalar_large ≈ copy(a4 .+ scalar_large)

  t5 = zeros(Float64, dims)
  a5 = copy(t5, mx.cpu())
  scalar_small = 1e-8
  scalar_large = 1e8
  @test t5 .+ scalar_small ≈ copy(a5 .+ scalar_small)
  @test t5 .+ scalar_large ≈ copy(a5 .+ scalar_large)

  t6 = zeros(Float16, dims)
  a6 = copy(t6, mx.cpu())
  scalar_small = Float16(1e-5)
  scalar_large = Float16(1e4)
  @test t6 .+ scalar_small ≈ copy(a6 .+ scalar_small)
  @test t6 .+ scalar_large ≈ copy(a6 .+ scalar_large)

  let x = mx.NDArray([1 2; 3 4]), y = mx.NDArray([1 1; 1 1])
    @test copy(42 .+ x) == [43 44; 45 46]
    @test copy(x .+ 42) == [43 44; 45 46]
    @test copy(0 .+ x .+ y .+ 41) == [43 44; 45 46]
  end

  @info("NDArray::plus::scalar::type convert")
  let x = mx.NDArray([1, 2, 3])
    y = x .+ 0.5
    @test copy(y) == copy(x)

    y = x .+ 2.9
    @test copy(y) == [3, 4, 5]
  end

  @info("NDArray::broadcast_add")
  let
    A = [1 2 3;
         4 5 6]
    B = [1,
         2]
    x = NDArray(A)
    y = NDArray(B)

    z = x .+ y
    @test copy(z) == A .+ B

    # TODO
    # @inplace x .+= y
    # @test copy(x) == A .+ B
  end
end

function test_minus()
  dims   = rand_dims()
  t1, a1 = rand_tensors(dims)
  t2, a2 = rand_tensors(dims)

  @info("NDArray::minus::dims = $dims")

  @test t1 .- t2 ≈ copy(a1 .- a2)

  @test -t1 ≈ copy(-a1)

  # make sure the negation is not in-place, so a1 is not changed after previous
  # statement is executed
  @test t1 ≈ copy(a1)

  # test inplace -= operation
  a0 = a1              # keep a reference to a1
  @mx.inplace a1 -= a2 # perform inplace -=
  @test a0 == a1       # make sure they are still the same object
  @test a0.handle == a1.handle
  @test copy(a0) ≈ copy(a1)
  @test copy(a1) ≈ t1 .- t2

  # test scalar
  scalar = rand()
  @test t2 .- scalar ≈ copy(a2 .- scalar)

  # test small and large scalar
  t4 = zeros(Float32, dims)
  a4 = copy(t4, mx.cpu())
  scalar_small = 1e-8
  scalar_large = 1e8
  @test t4 .- scalar_small ≈ copy(a4 .- scalar_small)
  @test t4 .- scalar_large ≈ copy(a4 .- scalar_large)

  t5 = zeros(Float64, dims)
  a5 = copy(t5, mx.cpu())
  scalar_small = 1e-8
  scalar_large = 1e8
  @test t5 .- scalar_small ≈ copy(a5 .- scalar_small)
  @test t5 .- scalar_large ≈ copy(a5 .- scalar_large)

  t6 = zeros(Float16, dims)
  a6 = copy(t6, mx.cpu())
  scalar_small = Float16(1e-5)
  scalar_large = Float16(1e4)
  @test t6 .- scalar_small ≈ copy(a6 .- scalar_small)
  @test t6 .- scalar_large ≈ copy(a6 .- scalar_large)

  @info("NDArray::minus::scalar::type convert")
  let x = mx.NDArray([1, 2, 3])
    @test copy(x .- π) ≈ [-2, -1, 0]
  end

  @info("NDArray::broadcast_minus")
  let
    A = [1 2 3;
         4 5 6]
    B = [1,
         2]
    x = NDArray(A)
    y = NDArray(B)

    z = x .- y
    @test copy(z) == A .- B

    # TODO
    # @inplace x .-= y
    # @test copy(x) == A .- B
  end

  @info("NDArray::scalar::rminus")
  let
    A = [1 2 3;
         4 5 6]
    B = 10 .- A

    x = NDArray(A)
    y = 10 .- x

    @test copy(y) == B
  end
end

function test_mul()
  dims   = rand_dims()
  t1, a1 = rand_tensors(dims)
  t2, a2 = rand_tensors(dims)
  t3, a3 = rand_tensors(dims)

  @info("NDArray::mul::dims = $dims")

  @test t1 .* t2 ≈ copy(a1 .* a2)

  # test inplace .*= operation
  a0 = a1               # keep a reference to a1
  @mx.inplace a1 .*= a2 # perform inplace .*=
  @test a0 == a1        # make sure they are still the same object
  @test a0.handle == a1.handle
  @test copy(a0) ≈ copy(a1)
  @test copy(a1) ≈ t1 .* t2

  # test scalar
  scalar = mx.MX_float(rand())
  @test t3 .* scalar ≈ copy(a3 .* scalar)

  # test small and large scalar
  t4, a4 = rand_tensors(Float32, dims)
  scalar_small = 1e-8
  scalar_large = 1e8
  @test t4 * scalar_small ≈ copy(a4 .* scalar_small)
  @test t4 * scalar_large ≈ copy(a4 .* scalar_large)

  t5, a5 = rand_tensors(Float64, dims)
  scalar_small = 1e-8
  scalar_large = 1e8
  @test t5 * scalar_small ≈ copy(a5 .* scalar_small)
  @test t5 * scalar_large ≈ copy(a5 .* scalar_large)

  t6, a6 = rand_tensors(Float16, dims)
  scalar_small = Float16(1e-5)
  @test t6 * scalar_small ≈ copy(a6 .* scalar_small)

  @info("NDArray::mul::matrix multiplication")
  let x = mx.NDArray([1.  2])
    y = x' * x
    @test copy(y) == [1. 2; 2 4]
  end

  @info("NDArray::mul::elementwise::issue 253")
  let x = mx.NDArray([1.  2])
    y = x .* x
    @test copy(y) == [1. 4.]
  end

  @info("NDArray::mul::scalar::type convert")
  let x = mx.NDArray([1, 2, 3])
    y = x .* π
    @test eltype(x) == Int
    @test copy(y) == [3, 6, 9]
  end

  @info("NDArray::broadcast_mul")
  let
    A = [1 2 3;
         4 5 6]
    B = [1,
         2]
    x = NDArray(A)
    y = NDArray(B)

    z = x .* y
    @test copy(z) == A .* B

    # TODO
    # @inplace x .*= y
    # @test copy(x) == A .* B
  end
end

function test_div()
  dims   = rand_dims()
  t1, a1 = rand_tensors(dims)
  t2, a2 = rand_tensors(dims)

  @info("NDArray::div::dims = $dims")
  t2             .+= 2  # avoid numerical instability
  @mx.inplace a2 .+= 2

  @test t1 ./ t2 ≈ copy(a1 ./ a2)

  # test inplace -= operation
  a0 = a1                # keep a reference to a2
  @mx.inplace a1 ./= a2  # perform inplace ./=
  @test a0 == a1         # make sure they are still the same object
  @test a0.handle == a1.handle
  @test copy(a0) ≈ copy(a1)
  @test copy(a1) ≈ t1 ./ t2

  # test scalar
  scalar = rand() + 2
  @test t2 ./ scalar ≈ copy(a2 ./ scalar)

  # test small and large scalar
  t4, a4 = rand_tensors(Float32, dims)
  scalar_small = 1e-8
  scalar_large = 1e8
  @test t4 ./ scalar_small ≈ copy(a4 ./ scalar_small)
  @test t4 ./ scalar_large ≈ copy(a4 ./ scalar_large)

  t5, a5 = rand_tensors(Float64, dims)
  scalar_small = 1e-8
  scalar_large = 1e8
  @test t5 ./ scalar_small ≈ copy(a5 ./ scalar_small)
  @test t5 ./ scalar_large ≈ copy(a5 ./ scalar_large)

  t6, a6 = rand_tensors(Float16, dims)
  scalar_large = 1e4
  @test t6 ./ scalar_large ≈ copy(a6 ./ scalar_large)

  @info("NDArray::div::scalar::type convert")
  let x = mx.NDArray([1, 2, 3])
    y = x ./ 1.1
    @test eltype(y) == Int
    @test copy(y) == [1, 2, 3]

    y = x ./ 2
    @test eltype(y) == Int  # this differs from julia
    @test copy(y) == [0, 1, 1]

    @test_throws AssertionError x ./ 0.5
  end

  @info("NDArray::broadcast_div")
  let
    A = Float32[1 2 3;
                4 5 6]
    B = Float32[1,
                2]
    x = NDArray(A)
    y = NDArray(B)

    z = x ./ y
    @test copy(z) == A ./ B

    # TODO
    # @inplace x ./= y
    # @test copy(x) == A ./ B
  end
end

function test_rdiv()
  @info("NDArray::rdiv")

  @info("NDArray::rdiv::Inf16")
  let x = 1 ./ mx.zeros(Float16, 4)
    @test copy(x) == [Inf16, Inf16, Inf16, Inf16]
  end

  @info("NDArray::rdiv::Inf32")
  let x = 1 ./ mx.zeros(Float32, 4)
    @test copy(x) == [Inf32, Inf32, Inf32, Inf32]
  end

  @info("NDArray::rdiv::Inf64")
  let x = 1 ./ mx.zeros(Float64, 4)
    @test copy(x) == [Inf64, Inf64, Inf64, Inf64]
  end

  @info("NDArray::rdiv::Int")
  let x = 1 ./ mx.NDArray([1 2; 3 4])
    @test copy(x) == [1 0; 0 0]
  end

  @info("NDArray::rdiv::Float32")
  let x = 1 ./ mx.NDArray(Float32[1 2; 3 4])
    y = 1 ./ Float32[1 2; 3 4]
    @test copy(x) ≈ y
  end

  @info("NDArray::rdiv::type convert")
  let x = mx.NDArray([1, 2, 3])
    y = 5.5 ./ x
    @test eltype(y) == Int  # this differs from julia
    @test copy(y) == [5, 2, 1]
  end
end  # function test_rdiv

function test_mod()
  @info("NDArray::mod")
  A = [1 2; 3 4]
  B = [1 1; 3 3]

  let x = NDArray(A), y = NDArray(B)
    C = A .% B
    D = B .% A

    w = x .% y
    z = y .% x

    @test copy(w) ≈ C
    @test copy(z) ≈ D
  end

  @info("NDArray::mod::scalar")
  let x = NDArray(A)
    C = A .% 2
    y = x .% 2
    @test copy(y) ≈ C
  end

  @info("NDArray::rmod")
  let x = NDArray(A)
    C = 11 .% A
    y = 11 .% x
    @test copy(y) ≈ C
  end

  @info("NDArray::mod_from!")
  let
    x = NDArray(A)
    y = NDArray(B)
    C = A .% B
    mx.mod_from!(x, y)
    @test copy(x) ≈ C
  end

  let
    x = NDArray(A)
    y = NDArray(B)
    C = B .% A
    mx.mod_from!(y, x)

    @test copy(y) ≈ C
  end

  @info("NDArray::mod_from!::scalar")
  let
    x = NDArray(A)
    C = A .% 2
    mx.mod_from!(x, 2)
    @test copy(x) ≈ C
  end

  @info("NDArray::rmod_from!")
  let
    x = NDArray(A)
    C = 11 .% A
    mx.rmod_from!(11, x)
    @test copy(x) ≈ C
  end

  @info("NDArray::mod_from!::writable")
  let
    x = NDArray(A)
    y = NDArray(B)
    x.writable = false
    y.writable = false
    @test_throws AssertionError mx.mod_from!(x, y)
    @test_throws AssertionError mx.mod_from!(y, x)
    @test_throws AssertionError mx.mod_from!(x, 2)
    @test_throws AssertionError mx.rmod_from!(2, x)
  end

  @info("NDArray::mod::inplace")
  let
    x = NDArray(A)
    y = NDArray(B)
    C = A .% B
    @inplace x .%= y
    @test copy(x) ≈ C
  end

  @info("NDArray::broadcast_mod")
  let
    A = [1 2 3;
         4 5 6]
    B = [1,
         2]
    x = NDArray(A)
    y = NDArray(B)

    z = x .% y
    @test copy(z) == A .% B

    # TODO
    # @inplace x .%= y
    # @test copy(x) == A .% B
  end
end  # function test_mod

function test_gd()
  dims   = rand_dims()
  tw, aw = rand_tensors(dims)
  tg, ag = rand_tensors(dims)

  @info("NDArray::gd::dims = $dims")

  lr = rand()
  wd = rand()

  @mx.inplace aw += -lr * (ag + wd * aw)
  tw += -lr * (tg + wd * tw)
  @test copy(aw) ≈ tw
end

function test_saveload()
  n_arrays = 5
  @info("NDArray::saveload::n_arrays = $n_arrays")
  fname = tempname()

  # save and load a single array
  dims   = rand_dims()
  j_array, nd_array = rand_tensors(dims)
  mx.save(fname, nd_array)
  data = mx.load(fname, mx.NDArray)
  @test data isa Vector{<:mx.NDArray}
  @test length(data) == 1
  @test copy(data[1]) ≈ j_array

  # save and load N arrays of different shape
  arrays = [rand_tensors(rand_dims()) for i = 1:n_arrays]
  nd_arrays = mx.NDArray[x[2] for x in arrays]
  mx.save(fname, nd_arrays)
  data = mx.load(fname, mx.NDArray)
  @test data isa Vector{<:mx.NDArray}
  @test length(data) == n_arrays
  for i = 1:n_arrays
    @test copy(data[i]) ≈ arrays[i][1]
  end

  # save and load dictionary of ndarrays
  names = [Symbol("array$i") for i = 1:n_arrays]
  dict = Dict([(n, v) for (n,v) in zip(names, nd_arrays)])
  mx.save(fname, dict)
  data = mx.load(fname, mx.NDArray)
  @test data isa Dict{Symbol,<:mx.NDArray}
  @test length(data) == n_arrays
  for i = 1:n_arrays
    @test copy(data[names[i]]) ≈ arrays[i][1]
  end

  rm(fname)
end

function test_clamp()
  @info("NDArray::clamp::dims")

  A = [1 2 3;
       4 5 6;
       7 8 9.]
  B = [3 3 3;
       4 5 6;
       7 8 8.]
  x = NDArray(A)
  y = clamp(x, 3., 8.)

  # make sure the original array is not modified
  @test copy(x) ≈ A
  @test copy(y) ≈ B

  @info("NDArray::clamp!")
  let
    x = NDArray(1.0:20)
    clamp!(x, 5, 15)
    @test all(5 .<= copy(x) .<= 15)
  end
end

function test_power()
  @info("NDArray::power")

  @info("NDArray::power::Int::x .^ n")
  let x = mx.NDArray([1 2; 3 4])
    @test eltype(x) == Int
    @test copy(x .^ -1)  == [1 0; 0 0]
    @test copy(x .^ 0)   == [1 1; 1 1]
    @test copy(x .^ 1)   == [1 2; 3 4]
    @test copy(x .^ 1.1) == [1 2; 3 4]
    @test copy(x .^ 2)   == [1 4; 9 16]
    @test copy(x .^ 2.9) == [1 4; 9 16]
    @test copy(x .^ 3)   == [1 8; 27 64]
  end

  @info("NDArray::power::Int::n .^ x")
  let x = mx.NDArray([1 2; 3 4])
    @test eltype(x) == Int
    @test copy(0   .^ x)   == [0 0; 0 0]
    @test copy(1   .^ x)   == [1 1; 1 1]
    @test copy(1.1 .^ x) == [1 1; 1 1]
    @test copy(2   .^ x)   == [2 4; 8 16]
    @test copy(2.9 .^ x) == [2 4; 8 16]
    @test copy(3   .^ x)   == [3 9; 27 81]
  end

  @info("NDArray::power::Int::x .^ y")
  let x = mx.NDArray([1 2; 3 4]), y = mx.NDArray([2 2; 2 2])
    @test eltype(x) == Int
    @test eltype(y) == Int
    @test copy(x .^ y) == [1 4; 9 16]
    @test copy(y .^ x) == [2 4; 8 16]
  end

  @info("NDArray::power::Float32::x .^ n")
  let x = mx.NDArray(Float32[1 2; 3 4]), A = Float32[1 2; 3 4]
    @test eltype(x) == Float32
    @test copy(x .^ 0) == Float32[1 1; 1 1]
    @test copy(x .^ 1) == Float32[1 2; 3 4]
    @test copy(x .^ 2) == Float32[1 4; 9 16]
    @test copy(x .^ 3) == Float32[1 8; 27 64]

    @test copy(x .^ -1)  ≈ A .^ -1
    @test copy(x .^ 1.1) ≈ A .^ 1.1
    @test copy(x .^ 2.9) ≈ A .^ 2.9
  end

  @info("NDArray::power::Float32::n .^ x")
  let x = mx.NDArray(Float32[1 2; 3 4]), A = Float32[1 2; 3 4]
    @test eltype(x) == Float32
    @test copy(0 .^ x) == Float32[0 0; 0 0]
    @test copy(1 .^ x) == Float32[1 1; 1 1]
    @test copy(2 .^ x) == Float32[2 4; 8 16]
    @test copy(3 .^ x) == Float32[3 9; 27 81]

    @test copy(1.1 .^ x) ≈ 1.1 .^ A
    @test copy(2.9 .^ x) ≈ 2.9 .^ A
  end

  @info("NDArray::power::Float32::x .^ y")
  let x = mx.NDArray(Float32[1 2; 3 4]), y = mx.NDArray(Float32[2 2; 2 2])
    @test eltype(x) == Float32
    @test eltype(y) == Float32
    @test copy(x .^ y) == Float32[1 4; 9 16]
    @test copy(y .^ x) == Float32[2 4; 8 16]
  end

  @info("NDArray::power::ℯ .^ x::x .^ ℯ")
  let x = mx.zeros(2, 3), A = [1 1 1; 1 1 1]
    @test copy(ℯ .^ x) ≈ A
  end

  let A = Float32[1 2; 3 4], x = mx.NDArray(A)
    @test copy(ℯ .^ x) ≈ ℯ .^ A
    @test copy(x .^ ℯ) ≈ A .^ ℯ
  end

  @info("NDArray::power::π .^ x::x .^ π")
  let A = Float32[1 2; 3 4], x = mx.NDArray(A)
    @test copy(π .^ x) ≈ π .^ A
    @test copy(x .^ π) ≈ A .^ π
  end

  # TODO: Float64: wait for https://github.com/apache/incubator-mxnet/pull/8012

  @info("NDArray::broadcast_power")
  let
    A = [1 2 3;
         4 5 6]
    B = [1,
         2]
    x = NDArray(A)
    y = NDArray(B)

    z = x.^y
    @test copy(z) == A.^B

    # TODO
    # @inplace x .^= y
    # @test copy(x) == A.^B
  end
end # function test_power

function test_sqrt()
  dims = rand_dims()
  @info("NDArray::sqrt::dims = $dims")

  j_array, nd_array = rand_tensors(dims)
  sqrt_ed = sqrt(nd_array)
  @test copy(sqrt_ed) ≈ sqrt.(j_array)
end

function test_nd_as_jl()
  dims = (2, 3)
  @info("NDArray::nd_as_jl::dims = $dims")

  x = mx.zeros(dims) + 5
  y = mx.ones(dims)
  z = mx.zeros(dims)
  @mx.nd_as_jl ro=x rw=(y, z) begin
    for i = 1:length(z)
      z[i] = x[i]
    end

    z[:, 1] = y[:, 1]
    y .= 0
  end

  @test sum(copy(y)) == 0
  @test sum(copy(z)[:, 1]) == 2
  @test copy(z)[:, 2:end] ≈ copy(x)[:, 2:end]
end

function test_dot()
  dims1 = (2, 3)
  dims2 = (3, 8)
  @info("NDArray::dot")

  x = mx.zeros(dims1)
  y = mx.zeros(dims2)
  z = mx.dot(x, y)
  @test size(z) == (2, 8)

  x = mx.zeros(1, 2)
  y = mx.zeros(1, 2, 3)
  @test_throws mx.MXError dot(x, y)  # dimension mismatch

  @info("NDArray::matrix mul")
  let
    A = [1. 2 3; 4 5 6]
    B = [-1., -2, -3]
    x = NDArray(A)
    y = NDArray(B)
    z = x * y
    @test copy(z) == A * B
    @test size(z) == (2,)
  end

  let
    A = [1. 2 3; 4 5 6]
    B = [-1. -2; -3 -4; -5 -6]
    x = NDArray(A)
    y = NDArray(B)
    z = x * y
    @test copy(z) == A * B
    @test size(z) == (2, 2)
  end
end

function test_eltype()
  @info("NDArray::eltype")
  dims = (3,3)

  x = NDArray(undef, dims)
  @test eltype(x) == mx.DEFAULT_DTYPE

  for TF in instances(mx.TypeFlag)
    T = mx.fromTypeFlag(TF)
    x = NDArray{T}(undef, dims)
    @test eltype(x) == T
  end
end

function test_reshape()
  @info("NDArray::reshape")
  A = rand(2, 3, 4)

  B = reshape(NDArray(A), 4, 3, 2)
  @test size(B) == (4, 3, 2)
  @test copy(B)[3, 1, 1] == A[1, 2, 1]

  C = reshape(NDArray(A), (4, 3, 2))
  @test size(C) == (4, 3, 2)
  @test copy(C)[3, 1, 1] == A[1, 2, 1]

  @info("NDArray::reshape::reverse")
  A = mx.zeros(10, 5, 4)

  B = reshape(A, -1, 0)
  @test size(B) == (40, 5)

  C = reshape(A, -1, 0, reverse=true)
  @test size(C) == (50, 4)
end

function test_expand_dims()
  @info("NDArray::expand_dims")
  let A = [1, 2, 3, 4], x = NDArray(A)
    @test size(x) == (4,)

    y = expand_dims(x, 1)
    @test size(y) == (1, 4)

    y = expand_dims(x, 2)
    @test size(y) == (4, 1)
  end

  let A = [1 2; 3 4; 5 6], x = NDArray(A)
    @test size(x) == (3, 2)

    y = expand_dims(x, 1)
    @test size(y) == (1, 3, 2)

    y = expand_dims(x, 2)
    @test size(y) == (3, 1, 2)

    y = expand_dims(x, 3)
    @test size(y) == (3, 2, 1)
  end
end  # test_expand_dims

function test_sum()
  @info("NDArray::sum")

  let A = reshape(1.0:8, 2, 2, 2), X = mx.NDArray(A)
    @test copy(sum(X))[]              == sum(A)
    @test copy(sum(X, dims = 1))      == sum(A, dims = 1)
    @test copy(sum(X, dims = 2))      == sum(A, dims = 2)
    @test copy(sum(X, dims = 3))      == sum(A, dims = 3)
    @test copy(sum(X, dims = [1, 2])) == sum(A, dims = [1, 2])
    @test copy(sum(X, dims = (1, 2))) == sum(A, dims = (1, 2))
  end
end

function test_mean()
  @info("NDArray::mean")

  let A = reshape(1.0:8, 2, 2, 2), X = mx.NDArray(A)
    @test copy(mean(X))[]              == mean(A)
    @test copy(mean(X, dims = 1))      == mean(A, dims = 1)
    @test copy(mean(X, dims = 2))      == mean(A, dims = 2)
    @test copy(mean(X, dims = 3))      == mean(A, dims = 3)
    @test copy(mean(X, dims = [1, 2])) == mean(A, dims = [1, 2])
    @test copy(mean(X, dims = (1, 2))) == mean(A, dims = (1, 2))
  end
end

function test_maximum()
  @info("NDArray::maximum")

  let A = reshape(1.0:8, 2, 2, 2), X = mx.NDArray(A)
    @test copy(maximum(X))[]              == maximum(A)
    @test copy(maximum(X, dims = 1))      == maximum(A, dims = 1)
    @test copy(maximum(X, dims = 2))      == maximum(A, dims = 2)
    @test copy(maximum(X, dims = 3))      == maximum(A, dims = 3)
    @test copy(maximum(X, dims = [1, 2])) == maximum(A, dims = [1, 2])
    @test copy(maximum(X, dims = (1, 2))) == maximum(A, dims = (1, 2))
  end

  @info("NDArray::broadcast_maximum")
  let
    A = [1 2 3;
         4 5 6]
    B = [1,
         2]
    x = NDArray(A)
    y = NDArray(B)

    z = max.(x, y)
    @test copy(z) == max.(A, B)
  end
end

function test_minimum()
  @info("NDArray::minimum")

  let A = reshape(1.0:8, 2, 2, 2), X = mx.NDArray(A)
    @test copy(minimum(X))[]              == minimum(A)
    @test copy(minimum(X, dims = 1))      == minimum(A, dims = 1)
    @test copy(minimum(X, dims = 2))      == minimum(A, dims = 2)
    @test copy(minimum(X, dims = 3))      == minimum(A, dims = 3)
    @test copy(minimum(X, dims = [1, 2])) == minimum(A, dims = [1, 2])
    @test copy(minimum(X, dims = (1, 2))) == minimum(A, dims = (1, 2))
  end

  @info("NDArray::broadcast_minimum")
  let
    A = [1 2 3;
         4 5 6]
    B = [1,
         2]
    x = NDArray(A)
    y = NDArray(B)

    z = min.(x, y)
    @test copy(z) == min.(A, B)
  end
end

function test_prod()
  @info("NDArray::prod")

  let A = reshape(1.0:8, 2, 2, 2), X = mx.NDArray(A)
    @test copy(prod(X))[]              == prod(A)
    @test copy(prod(X, dims = 1))      == prod(A, dims = 1)
    @test copy(prod(X, dims = 2))      == prod(A, dims = 2)
    @test copy(prod(X, dims = 3))      == prod(A, dims = 3)
    @test copy(prod(X, dims = [1, 2])) == prod(A, dims = [1, 2])
    @test copy(prod(X, dims = (1, 2))) == prod(A, dims = (1, 2))
  end
end

function test_fill()
  @info("NDArray::fill")

  let x = mx.fill(42, 2, 3, 4)
    @test eltype(x) == Int
    @test size(x) == (2, 3, 4)
    @test copy(x) == fill(42, 2, 3, 4)
  end

  let x = mx.fill(Float32(42), 2, 3, 4)
    @test eltype(x) == Float32
    @test size(x) == (2, 3, 4)
    @test copy(x) ≈ fill(Float32(42), 2, 3, 4)
  end

  let x = mx.fill(42, (2, 3, 4))
    @test eltype(x) == Int
    @test size(x) == (2, 3, 4)
    @test copy(x) == fill(42, 2, 3, 4)
  end

  let x = mx.fill(Float32(42), (2, 3, 4))
    @test eltype(x) == Float32
    @test size(x) == (2, 3, 4)
    @test copy(x) ≈ fill(Float32(42), 2, 3, 4)
  end

  @info("NDArray::fill!::arr")
  let x = fill!(mx.zeros(2, 3, 4), 42)
    @test eltype(x) == Float32
    @test size(x) == (2, 3, 4)
    @test copy(x) ≈ fill(Float32(42), 2, 3, 4)
  end
end  # function test_fill

function test_transpose()
  @info("NDArray::transpose::1D")
  let A = rand(Float32, 4), x = NDArray(A)
    @test size(x) == (4,)
    @test size(x') == (1, 4)
  end

  @info("NDArray::transpose::2D")
  let A = rand(Float32, 2, 3), x = mx.NDArray(A)
    @test size(x) == (2, 3)
    @test size(x') == (3, 2)
  end

  @info("NDArray::permutedims")
  let A = collect(Float32, reshape(1.0:24, 2, 3, 4)), x = mx.NDArray(A)
    A′ = permutedims(A, [2, 1, 3])
    x′ = permutedims(x, [2, 1, 3])
    @test size(A′) == size(x′)
    @test A′ == copy(x′)
  end
end

function test_show()
  @info("NDArray::show::REPL")
  let str = sprint(show, MIME"text/plain"(), mx.NDArray([1 2 3 4]))
    @test occursin("1×4", str)
    @test occursin("NDArray", str)
    @test occursin("Int64", str)
    @test occursin("CPU", str)
    @test match(r"1\s+2\s+3\s+4", str) != nothing
  end

  @info("NDArray::show")
  let str = sprint(show, mx.NDArray([1 2 3 4]))
    @test str == "NDArray([1 2 3 4])"
  end

  let str = sprint(show, mx.zeros(4))
    @test str == "NDArray(Float32[0.0, 0.0, 0.0, 0.0])"
  end
end

function test_size()
  @info("NDArray::size")
  let A = [1 2; 3 4; 5 6], x = mx.NDArray(A)
    @test size(A) == size(x)
    dims = (1, 2, 3, 4, 5)
    @test map(d -> size(A, d), dims) == map(d -> size(x, d), dims)
    @inferred map(d -> size(x, d), dims)
  end
end  # function test_size()

function check_trigonometric(f)
  @info("NDArray::$f")
  let A = [.1 .2; .3 .4], x = mx.NDArray(A)
    B = f.(A)
    y = f.(x)
    @test copy(y) ≈ B
  end

  let A = Float32[.1 .2; .3 .4], x = mx.NDArray(A)
    B = f.(A)
    y = f.(x)
    @test copy(y) ≈ B
  end
end  # function check_trigonometric

function test_trigonometric()
  for f ∈ [sin, cos, tan, asin, acos, atan]
    check_trigonometric(f)
  end
end  # function test_trigonometric

function check_hyperbolic(f, A)
  @info("NDArray::$f")
  let x = NDArray(A)
    B = f.(A)
    y = f.(x)
    @test copy(y) ≈ B
  end

  let A = Float32.(A), x = NDArray(A)
    B = f.(A)
    y = f.(x)
    @test copy(y) ≈ B
  end
end  # function check_hyperbolic

function test_hyperbolic()
  for f ∈ [sinh, cosh, tanh, asinh, acosh, atanh]
    A = if f == acosh
      [1.1, 1.2, 1.3, 1.4]
    else
      [.1, .2, .3, .4]
    end
    check_hyperbolic(f, A)
  end
end  # function test_hyperbolic

function test_act_funcs()
  @info("NDArray::σ/sigmoid")
  let
    A = Float32[.1, .2, -.3, -.4]
    B = @. 1 / (1 + ℯ ^ (-A))
    x = NDArray(A)
    y = σ.(x)
    @test copy(y) ≈ B

    z = sigmoid.(x)
    @test copy(z) ≈ B
  end

  @info("NDArray::relu")
  let
    A = [1, 2, -3, -4]
    B = max.(A, 0)
    x = NDArray(A)
    y = relu.(x)
    @test copy(y) ≈ B
  end

  @info("NDArray::softmax::1D")
  let
    A = Float32[1, 2, 3, 4]
    B = exp.(A) ./ sum(exp.(A))
    x = NDArray(A)
    y = softmax.(x)
    @test copy(y) ≈ B
  end

  @info("NDArray::softmax::2D")
  let
    A = Float32[1 2; 3 4]
    B = exp.(A) ./ sum(exp.(A), dims = 1)
    x = NDArray(A)
    y = softmax.(x, 1)
    @test copy(y) ≈ B

    C = exp.(A) ./ sum(exp.(A), dims = 2)
    z = softmax.(x, 2)
    @test copy(z) ≈ C
  end

  @info("NDArray::log_softmax::1D")
  let
    A = Float32[1, 2, 3, 4]
    B = log.(exp.(A) ./ sum(exp.(A)))
    x = NDArray(A)
    y = log_softmax.(x)
    @test copy(y) ≈ B
  end

  @info("NDArray::log_softmax::2D")
  let
    A = Float32[1 2; 3 4]
    B = log.(exp.(A) ./ sum(exp.(A), dims = 1))
    x = NDArray(A)
    y = log_softmax.(x, 1)
    @test copy(y) ≈ B

    C = log.(exp.(A) ./ sum(exp.(A), dims = 2))
    z = log_softmax.(x, 2)
    @test copy(z) ≈ C
  end
end  # function test_act_funcs

macro check_equal(op)
  quote
    A = [1 2 3
         4 5 6]
    B = [1,
         6]
    x = NDArray(A)
    y = NDArray(B)
    a = broadcast($op, x, y)
    @test copy(a) == broadcast($op, A, B)

    C = [3 2 1
         6 5 4]
    z = NDArray(C)
    b = broadcast($op, x, z)
    @test copy(b) == broadcast($op, A, C)
  end
end

function test_equal()
  @info("NDArray::broadcast_equal")
  @check_equal ==

  @info("NDArray::broadcast_not_equal")
  @check_equal !=

  @info("NDArray::broadcast_greater")
  @check_equal >

  @info("NDArray::broadcast_greater_equal")
  @check_equal >=

  @info("NDArray::broadcast_lesser")
  @check_equal <

  @info("NDArray::broadcast_lesser_equal")
  @check_equal <=
end  # function test_equal

function test_broadcast_to()
  @info("NDArray::broadcast_to")
  A = [1 2 3]
  x = NDArray(A)
  @test mx.broadcast_to(x, (1, 3)) |> copy == A
  @test mx.broadcast_to(x, (5, 3)) |> copy == repeat(A, outer = (5, 1))

  @test mx.broadcast_to(x, 1, 3) |> copy == A
  @test mx.broadcast_to(x, 5, 3) |> copy == repeat(A, outer = (5, 1))
end  # function test_broadcast_to

function test_broadcast_axis()
  @info("NDArray::broadcast_axis")
  A = reshape([1, 2, 3], 1, 3, 1)
  x = NDArray(A)

  @test mx.broadcast_axis(x, 1, 4) |> copy == [A; A; A; A]
  @test mx.broadcast_axis(x, 3, 2) |> copy == cat(A, A, dims = 3)

  @info("NDArray::broadcast_axes")
  @test mx.broadcast_axes(x, 1, 4) |> copy == [A; A; A; A]
  @test mx.broadcast_axes(x, 3, 2) |> copy == cat(A, A, dims = 3)
end  # function test_broadcast_axis

function test_hypot()
  @info("NDArray::hypot")
  A = [3 3 3]
  B = [4, 4]
  C = hypot.(A, B)

  x = NDArray(A)
  y = NDArray(B)
  z = hypot.(x, y)

  @test copy(z) == C
end  # function test_hypot

function test_argmax()
  @info "NDArray::argmax"
  let
    A = [1. 5 3;
         4 2 6]
    x = NDArray(A)

    @test copy(argmax(x, dims = 1)) == [x[1] for x ∈ argmax(A, dims = 1)]
    @test copy(argmax(x, dims = 2)) == [x[2] for x ∈ argmax(A, dims = 2)]
  end

  @info "NDArray::argmax::NaN"
  let
    A = [1.  5 3;
         NaN 2 6]
    x = NDArray(A)

    @test copy(argmax(x, dims = 1)) == [x[1] for x ∈ argmax(A, dims = 1)]
    @test copy(argmax(x, dims = 2)) == [x[2] for x ∈ argmax(A, dims = 2)]
  end
end

function test_argmin()
  @info "NDArray::argmin"
  let
    A = [1. 5 3;
         4 2 6]
    x = NDArray(A)

    @test copy(argmin(x, dims = 1)) == [x[1] for x ∈ argmin(A, dims = 1)]
    @test copy(argmin(x, dims = 2)) == [x[2] for x ∈ argmin(A, dims = 2)]
  end

  @info "NDArray::argmin::NaN"
  let
    A = [1.  5 3;
         NaN 2 6]
    x = NDArray(A)

    @test copy(argmin(x, dims = 1)) == [x[1] for x ∈ argmin(A, dims = 1)]
    @test copy(argmin(x, dims = 2)) == [x[2] for x ∈ argmin(A, dims = 2)]
  end
end

################################################################################
# Run tests
################################################################################
@testset "NDArray Test" begin
  test_constructor()
  test_ones_zeros_like()
  test_assign()
  test_copy()
  test_slice()
  test_linear_idx()
  test_first()
  test_lastindex()
  test_cat()
  test_plus()
  test_minus()
  test_mul()
  test_div()
  test_rdiv()
  test_mod()
  test_gd()
  test_saveload()
  test_clamp()
  test_power()
  test_sqrt()
  test_eltype()
  test_nd_as_jl()
  test_dot()
  test_reshape()
  test_expand_dims()
  test_sum()
  test_mean()
  test_maximum()
  test_minimum()
  test_prod()
  test_fill()
  test_transpose()
  test_show()
  test_size()
  test_trigonometric()
  test_hyperbolic()
  test_act_funcs()
  test_equal()
  test_broadcast_to()
  test_broadcast_axis()
  test_hypot()
  test_argmax()
  test_argmin()
end

end
