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
  if @compat(is_windows())
    info("Please follow the libmxnet documentation on how to build manually")
    info("or to install pre-build packages:")
    info("http://mxnet.readthedocs.io/en/latest/how_to/build.html#building-on-windows")
    error("Automatic building libxmnet on Windows is currently not supported yet.")
  end

  openblas_path = Libdl.dlpath(Libdl.dlopen(Base.libblas_name))

  if VERSION >= v"0.5.0-dev+4338"
    blas_vendor = Base.BLAS.vendor()
  else
    blas_vendor = Base.blas_vendor()
  end

  ilp64 = ""
  if blas_vendor == :openblas64
    ilp64 = "-DINTERFACE64"
  end

  #--------------------------------------------------------------------------------
  # Build libmxnet
  mxnet = library_dependency("mxnet", aliases=["libmxnet", "libmxnet.so"])

  _prefix = joinpath(BinDeps.depsdir(mxnet), "usr")
  _srcdir = joinpath(BinDeps.depsdir(mxnet),"src")
  _mxdir  = joinpath(_srcdir, "mxnet")
  _libdir = joinpath(_prefix, "lib")
  provides(BuildProcess,
    (@build_steps begin
      CreateDirectory(_srcdir)
      CreateDirectory(_libdir)
      @build_steps begin
        ChangeDirectory(_srcdir)
        `rm -rf mxnet`
        `git clone --recursive https://github.com/dmlc/mxnet`
        @build_steps begin
          ChangeDirectory(joinpath(_srcdir, "mxnet"))
          `git checkout $libmxnet_curr_ver`
        end
        FileRule(joinpath(_libdir, "libmxnet.so"), @build_steps begin
          ChangeDirectory(_mxdir)
          `cp make/config.mk config.mk`
          if is_apple()
            `cp make/osx.mk config.mk`
          end
          `sed -i -s 's/USE_OPENCV = 1/USE_OPENCV = 0/' config.mk`
          `sed -i -s "s/MSHADOW_CFLAGS = \(.*\)/MSHADOW_CFLAGS = \1 $ilp64/" mshadow/make/mshadow.mk`
          `cp ../../cblas.h include/cblas.h`
          `make USE_BLAS=openblas MSHADOW_LDFLAGS="$openblas_path"`
          `cp lib/libmxnet.so $_libdir`
        end)
      end
    end), mxnet)

  @BinDeps.install Dict(:mxnet => :mxnet)
end
