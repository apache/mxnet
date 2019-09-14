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

using CMake
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
const USE_JEMALLOC = get(ENV, "USE_JEMALLOC", nothing)  # "ON" or "OFF"

get_cpucore() = min(Sys.CPU_THREADS, 32)

cmake_bool(x::Bool) = ifelse(x ≡ true, "ON", "OFF")

cmake_jemalloc(::Nothing) = ""
cmake_jemalloc(x::Bool)   =  "-DUSE_JEMALLOC=" * cmake_bool(x)

cmake_cuda_path(::Nothing) = ""
cmake_cuda_path(x::String) = "-DUSE_CUDA_PATH=" * x

cmake_jl_blas(x::Bool, blas_path) = ifelse(x, "-DOpenBLAS_LIB=$blas_path", "")

using BinDeps
@BinDeps.setup
if !libmxnet_detected
  if Sys.iswindows()
    if Sys.ARCH != :x86_64
      @info("Prebuilt windows binaries are only available on 64bit. You will have to built MXNet yourself.")
      return
    end
    @info("Downloading pre-built packages for Windows.")
    base_url = "https://github.com/yajiedesign/mxnet/releases/download/weekly_binary_build_v2/prebuildbase_win10_x64_vc14_v2.7z"

    if libmxnet_curr_ver == "master"
      _cmd = "{
        [System.Net.ServicePointManager]::SecurityProtocol='tls12';
        Invoke-WebRequest -Uri 'https://api.github.com/repos/yajiedesign/mxnet/releases/latest'
        -OutFile 'mxnet.json'}"
      # download_cmd uses powershell 2, but we need powershell 3 to do this
      run(`powershell -NoProfile -Command $_cmd`)
      curr_win = JSON.parsefile("mxnet.json")["tag_name"]
      @info("Can't use MXNet master on Windows, using latest binaries from $curr_win.")
    end
    # TODO: Get url from JSON.
    # TODO: detect cuda version and select corresponding url.
    name = "mxnet_x64_$(HAS_CUDA ? "vc141_gpu_cu101" : "vc14_cpu").7z"
    package_url = "https://github.com/yajiedesign/mxnet/releases/download/$(curr_win)/$(curr_win)_$(name)"

    exe7z = joinpath(Sys.BINDIR, "7z.exe")

    run(download_cmd(package_url, "mxnet.7z"))
    # this command will create the dir "usr\\lib"
    run(`$exe7z e mxnet.7z *\\build\\* *\\lib\\* -y -ousr\\lib`)

    run(download_cmd(base_url, "mxnet_base.7z"))
    run(`$exe7z x mxnet_base.7z -y -ousr`)
    run(`cmd /c copy "usr\\prebuildbase_win10_x64_vc14_v2\\3rdparty\\bin\\*.dll" "usr\\lib"`)

    # testing
    run(`cmd /c dir "usr\\lib"`)
    return
  end  # if Sys.iswindows()

  ################################################################################
  # If not found, try to build automatically using BinDeps
  ################################################################################

  blas_path = Libdl.dlpath(Libdl.dlopen(Base.libblas_name))
  blas_vendor = LinearAlgebra.BLAS.vendor()

  if blas_vendor == :unknown
    @info("Julia is built with an unkown blas library ($blas_path).")
    @info("Attempting build without reusing the blas library")
    USE_JULIA_BLAS = false
  elseif !(blas_vendor in (:openblas, :openblas64))
    @info("Unsure if we can build against $blas_vendor.")
    @info("Attempting build anyway.")
    USE_JULIA_BLAS = true
  else
    USE_JULIA_BLAS = false
  end
  @info "USE_JULIA_BLAS -> $USE_JULIA_BLAS"

  blas_name = occursin("openblas", string(blas_vendor)) ?  "open" : string(blas_vendor)

  #--------------------------------------------------------------------------------
  # Build libmxnet
  mxnet = library_dependency("mxnet", aliases=["mxnet", "libmxnet", "libmxnet.so"])

  _blddir = joinpath(BinDeps.depsdir(mxnet), "build")
  _srcdir = joinpath(BinDeps.depsdir(mxnet), "src")
  _mxdir  = joinpath(_srcdir, "mxnet")

  # We have do eagerly delete the build stuffs.
  # Otherwise we won't rebuild on an update.
  rm(_blddir, recursive=true, force=true)

  @debug "build dir -> $_blddir"

  provides(BuildProcess,
    (@build_steps begin
      CreateDirectory(_blddir)
      CreateDirectory(_srcdir)
      @build_steps begin
        BinDeps.DirectoryRule(_mxdir, @build_steps begin
          ChangeDirectory(_srcdir)
          `git clone --recursive https://github.com/apache/incubator-mxnet mxnet`
        end)
        @build_steps begin
          ChangeDirectory(_mxdir)
          `git fetch`
          `git submodule update`
          if libmxnet_curr_ver != "master"
            `git checkout $libmxnet_curr_ver`
          else
            `git checkout origin/$libmxnet_curr_ver`
          end
          `cp -f -v ../../cblas.h include/cblas.h`
        end
        @build_steps begin
          ChangeDirectory(_blddir)
          `$cmake
            -DCMAKE_BUILD_TYPE=$(libmxnet_curr_ver == "master" ? "Debug" : "Release")
            -DUSE_BLAS=$blas_name
            -DUSE_OPENCV=$(cmake_bool(false))
            -DUSE_CUDA=$(cmake_bool(HAS_CUDA))
            -DUSE_CUDNN=$(cmake_bool(HAS_CUDNN))
            $(cmake_jemalloc(USE_JEMALLOC))
            $(cmake_cuda_path(get(ENV, "CUDA_HOME", nothing)))
            $(cmake_jl_blas(USE_JULIA_BLAS, blas_path))
            $_mxdir`
          `make -j$(get_cpucore()) VERBOSE=$(Int(libmxnet_curr_ver == "master"))`
        end
        FileRule(joinpath(_blddir, "libmxnet.$(Libdl.dlext)"), @build_steps begin
          # the output file on macos is still in `.so` suffix,
          # so we create a soft link for it.
          `ln -s libmxnet.so $_blddir/libmxnet.$(Libdl.dlext)`
        end)
      end
    end), mxnet, installed_libpath=_blddir)

  @BinDeps.install Dict(:mxnet => :mxnet)
end
