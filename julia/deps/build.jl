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

using JSON
using Libdl
using LinearAlgebra

################################################################################
# First try to detect and load existing libmxnet
################################################################################
libmxnet_detected = false
libmxnet_curr_ver = get(ENV, "MXNET_COMMIT", "master")
curr_win = "20190608"  # v1.5.0

if haskey(ENV, "MXNET_HOME")
  MXNET_HOME = ENV["MXNET_HOME"]
  @info("MXNET_HOME environment detected: $MXNET_HOME")
  @info("Trying to load existing libmxnet...")
  # In case of macOS, if user build libmxnet from source and set the MXNET_HOME,
  # the output is still named as `libmxnet.so`.
  lib = Libdl.find_library(["libmxnet.$(Libdl.dlext)", "libmxnet.so"],
                           [joinpath(MXNET_HOME, "lib"), MXNET_HOME])
  if !isempty(lib)
    @info("Existing libmxnet detected at $lib, skip building...")
    libmxnet_detected = true
  else
    @info("Failed to load existing libmxnet, trying to build from source...")
  end
end

# Try to find cuda
CUDAPATHS = String[]
if haskey(ENV, "CUDA_HOME")
  push!(CUDAPATHS, joinpath(ENV["CUDA_HOME"], "lib64"))
elseif Sys.islinux()
  append!(CUDAPATHS, ["/opt/cuda/lib64", "/usr/local/cuda/lib64"])
end

if Sys.isunix()
  nvcc_path = Sys.which("nvcc")
  if nvcc_path ≢ nothing
    @info "Found nvcc: $nvcc_path"
    push!(CUDAPATHS, replace(nvcc_path, "bin/nvcc" => "lib64"))
  end
end

HAS_CUDA = false
HAS_CUDNN = false
let cudalib = Libdl.find_library(["libcuda", "nvcuda.dll"], CUDAPATHS)
  global HAS_CUDA = !isempty(cudalib) && Libdl.dlopen_e(cudalib) != C_NULL
end

if !HAS_CUDA && Sys.iswindows()
  # TODO: this needs to be improved.
  try
    run(`nvcc --version`)
    global HAS_CUDA = true
  catch
  end
end

if HAS_CUDA  # then check cudnn
  let cudnnlib = Libdl.find_library("libcudnn", CUDAPATHS)
    global HAS_CUDNN = !isempty(cudnnlib) && Libdl.dlopen_e(cudnnlib) != C_NULL
    if HAS_CUDNN && !haskey(ENV, "CUDA_HOME")  # inference `CUDA_HOME`
      ENV["CUDA_HOME"] = dirname(dirname(cudnnlib))
    end
  end
end

if HAS_CUDA
  @info("Found a CUDA installation.")
  if HAS_CUDNN
    @info("Found a CuDNN installation.")
  end
  @info("CUDA_HOME -> $(get(ENV, "CUDA_HOME", "nothing"))")
else
  @info("Did not find a CUDA installation, using CPU-only version of MXNet.")
end

# propagate more build flags from ENV
const CC  = get(ENV, "CC", nothing)
const CXX = get(ENV, "CXX", nothing)
const ADD_CFLAGS  = get(ENV, "ADD_CFLAGS", nothing)
const ADD_LDFLAGS = get(ENV, "ADD_LDFLAGS", nothing)
const USE_JEMALLOC = get(ENV, "USE_JEMALLOC", nothing)  # "0" or "1"

function get_cpucore()
    if haskey(ENV, "TRAVIS")  # on travis-ci
        2
    else
        min(Sys.CPU_THREADS, 32)
    end
end

using BinDeps
@BinDeps.setup
if !libmxnet_detected
  if Sys.iswindows()
    @info("Prebuilt windows binaries is not available currently. You will have to built MXNet yourself.")
    return
  end

  ################################################################################
  # If not found, try to build automatically using BinDeps
  ################################################################################

  blas_path = Libdl.dlpath(Libdl.dlopen(Base.libblas_name))
  blas_vendor = LinearAlgebra.BLAS.vendor()

  ilp64 = ""
  if blas_vendor == :openblas64
    ilp64 = "-DINTERFACE64"
  end

  FORCE_LAPACK = false
  if blas_vendor == :unknown
    @info("Julia is built with an unkown blas library ($blas_path).")
    @info("Attempting build without reusing the blas library")
    USE_JULIA_BLAS = false
  elseif !(blas_vendor in (:openblas, :openblas64))
    @info("Unsure if we can build against $blas_vendor.")
    @info("Attempting build anyway.")
    USE_JULIA_BLAS = true
  else
    USE_JULIA_BLAS = true
    FORCE_LAPACK = true
  end
  @info("USE_JULIA_BLAS -> $USE_JULIA_BLAS")

  blas_name = blas_vendor == :openblas64 ? "openblas" : string(blas_vendor)
  MSHADOW_LDFLAGS = "MSHADOW_LDFLAGS=-lm $blas_path"

  #--------------------------------------------------------------------------------
  # Build libmxnet
  mxnet = library_dependency("mxnet", aliases=["mxnet", "libmxnet", "libmxnet.so"])

  _prefix = joinpath(BinDeps.depsdir(mxnet), "usr")
  _srcdir = joinpath(BinDeps.depsdir(mxnet), "src")
  _mxdir  = joinpath(_srcdir, "mxnet")
  _libdir = joinpath(_prefix, "lib")
  # We have do eagerly delete the installed libmxnet.so
  # Otherwise we won't rebuild on an update.
  run(`rm -f $_libdir/libmxnet.$(Libdl.dlext)`)
  provides(BuildProcess,
    (@build_steps begin
      CreateDirectory(_srcdir)
      CreateDirectory(_libdir)
      @build_steps begin
        BinDeps.DirectoryRule(_mxdir, @build_steps begin
          ChangeDirectory(_srcdir)
          `git clone https://github.com/apache/mxnet mxnet`
        end)
        @build_steps begin
          ChangeDirectory(_mxdir)
          `git fetch`
          if libmxnet_curr_ver != "master"
            `git checkout $libmxnet_curr_ver`
          else
            `git checkout origin/$libmxnet_curr_ver`
          end
          `git submodule update --init --recursive`
          `git -C 3rdparty/mshadow checkout -- make/mshadow.mk`
          `cp -v ../../cblas.h include/cblas.h`
          `sed -i -s "s/MSHADOW_CFLAGS = \(.*\)/MSHADOW_CFLAGS = \1 $ilp64/" 3rdparty/mshadow/make/mshadow.mk`

          # Copy config.mk, always override the file
          if Sys.isapple()
            `cp make/osx.mk config.mk`
          else
            `cp make/config.mk config.mk`
          end

          # Configure OpenCV
          `sed -i -s 's/USE_OPENCV = 1/USE_OPENCV = 0/' config.mk`

          # Configure CUDA
          if HAS_CUDA
            @build_steps begin
              `sed -i -s 's/USE_CUDA = 0/USE_CUDA = 1/' config.mk`
              # address https://github.com/apache/mxnet/pull/7856
              `sed -i -s "s/ADD_LDFLAGS =\(.*\)/ADD_LDFLAGS =\1 -lcublas -lcusolver -lcurand -lcudart/" config.mk`
              if haskey(ENV, "CUDA_HOME")
                `sed -i -s "s@USE_CUDA_PATH = NONE@USE_CUDA_PATH = $(ENV["CUDA_HOME"])@" config.mk`
              end
              if haskey(ENV, "CUDA_HOME")
                # address https://github.com/apache/mxnet/pull/7838
                flag = "-L$(ENV["CUDA_HOME"])/lib64 -L$(ENV["CUDA_HOME"])/lib"
                `sed -i -s "s@ADD_LDFLAGS =\(.*\)@ADD_LDFLAGS =\1 $flag@" config.mk`
              end
              if HAS_CUDNN
                `sed -i -s 's/USE_CUDNN = 0/USE_CUDNN = 1/' config.mk`
              end
            end
          end

          # Force enable LAPACK build
          # Julia's OpenBLAS has LAPACK functionality already
          if FORCE_LAPACK
            if Sys.isapple()
              MSHADOW_LDFLAGS *= " -framework Accelerate"
            end
            `sed -i -s 's/ADD_CFLAGS =\(.*\)/ADD_CFLAGS =\1 -DMXNET_USE_LAPACK/' config.mk`
          end

          # propagate more build flags from ENV
          if CC != nothing
            `sed -i -s "s@^export CC =\(.*\)@export CC = $CC@" config.mk`
          end
          if CXX != nothing
            `sed -i -s "s@^export CXX =\(.*\)@export CXX = $CXX@" config.mk`
          end
          if ADD_CFLAGS != nothing
            `sed -i -s "s@ADD_CFLAGS =\(.*\)@ADD_CFLAGS =\1 $ADD_CFLAGS@" config.mk`
          end
          if ADD_LDFLAGS != nothing
            `sed -i -s "s@ADD_LDFLAGS =\(.*\)@ADD_LDFLAGS =\1 $ADD_LDFLAGS@" config.mk`
          end
          if USE_JEMALLOC != nothing
            `sed -i -s "s@USE_JEMALLOC =\(.*\)@USE_JEMALLOC = $USE_JEMALLOC@" config.mk`
          end

          if USE_JULIA_BLAS
            `make -j$(get_cpucore()) USE_BLAS=$blas_name $MSHADOW_LDFLAGS`
          else
            `make -j$(get_cpucore())`
          end
        end
        FileRule(joinpath(_libdir, "libmxnet.$(Libdl.dlext)"), @build_steps begin
          # the output file on macos is still in `.so` suffix
          # so we rename it
          `cp $_mxdir/lib/libmxnet.so $_libdir/libmxnet.$(Libdl.dlext)`
        end)
      end
    end), mxnet, installed_libpath=_libdir)

  @BinDeps.install Dict(:mxnet => :mxnet)
end
