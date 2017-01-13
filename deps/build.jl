using Compat

################################################################################
# First try to detect and load existing libmxnet
################################################################################
libmxnet_detected = false
libmxnet_curr_ver = "master"

if haskey(ENV, "MXNET_HOME")
  info("MXNET_HOME environment detected: $(ENV["MXNET_HOME"])")
  info("Trying to load existing libmxnet...")
  lib = Libdl.find_library(["libmxnet", "libmxnet.so"], ["$(ENV["MXNET_HOME"])/lib"])
  if !isempty(lib)
    info("Existing libmxnet detected at $lib, skip building...")
    libmxnet_detected = true
  else
    info("Failed to load existing libmxnet, trying to build from source...")
  end
end

using BinDeps
@BinDeps.setup
if !libmxnet_detected
  ################################################################################
  # If not found, try to build automatically using BinDeps
  ################################################################################
  if is_windows()
    info("Please follow the libmxnet documentation on how to build manually")
    info("or to install pre-build packages:")
    info("http://mxnet.readthedocs.io/en/latest/how_to/build.html#building-on-windows")
    error("Automatic building libxmnet on Windows is currently not supported yet.")
  end

  blas_path = Libdl.dlpath(Libdl.dlopen(Base.libblas_name))

  if VERSION >= v"0.5.0-dev+4338"
    blas_vendor = Base.BLAS.vendor()
  else
    blas_vendor = Base.blas_vendor()
  end

  ilp64 = ""
  if blas_vendor == :openblas64
    ilp64 = "-DINTERFACE64"
  end

  if blas_vendor == :unknown
    info("Julia is built with an unkown blas library ($blas_path).")
    info("Attempting build without reusing the blas library")
    USE_JULIA_BLAS = false
  elseif !(blas_vendor in (:openblas, :openblas64))
    info("Unsure if we can build against $blas_vendor.")
    info("Attempting build anyway.")
    USE_JULIA_BLAS = true
  else
    USE_JULIA_BLAS = true
  end

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
  run(`rm -f $_libdir/libmxnet.so`)
  provides(BuildProcess,
    (@build_steps begin
      CreateDirectory(_srcdir)
      CreateDirectory(_libdir)
      @build_steps begin
        BinDeps.DirectoryRule(_mxdir, @build_steps begin
          ChangeDirectory(_srcdir)
          `git clone --recursive https://github.com/dmlc/mxnet`
        end)
        @build_steps begin
          ChangeDirectory(_mxdir)
          # TODO(vchuravy). We have to reset mshadow/make/mshadow.mk
          `git -C mshadow checkout -- make/mshadow.mk`
          `git fetch`
          `git checkout $libmxnet_curr_ver`
          `git submodule update`
          `sed -i -s "s/MSHADOW_CFLAGS = \(.*\)/MSHADOW_CFLAGS = \1 $ilp64/" mshadow/make/mshadow.mk`
        end
        FileRule(joinpath(_mxdir, "config.mk"), @build_steps begin
          ChangeDirectory(_mxdir)
          if is_apple()
            `cp make/osx.mk config.mk`
          else
            `cp make/config.mk config.mk`
          end
          `sed -i -s 's/USE_OPENCV = 1/USE_OPENCV = 0/' config.mk`
        end)
        @build_steps begin
          ChangeDirectory(_mxdir)
          `cp ../../cblas.h include/cblas.h`
          if USE_JULIA_BLAS
            `make -j$(nprocs()) USE_BLAS=$blas_name $MSHADOW_LDFLAGS`
          else
            `make -j$(nprocs())`
          end
        end
        FileRule(joinpath(_libdir, "libmxnet.so"), @build_steps begin
          `cp $_mxdir/lib/libmxnet.so $_libdir/`
        end)
      end
    end), mxnet, installed_libpath=_libdir)

  @BinDeps.install Dict(:mxnet => :mxnet)
end
