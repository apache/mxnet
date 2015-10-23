################################################################################
# First try to detect and load existing libmxnet
################################################################################
# if haskey(ENV, "MXNET_HOME")
#   info("MXNET_HOME environment detected: $(ENV["MXNET_HOME"])")
#   info("Trying to load existing libmxnet...")
#   lib = Libdl.find_library(["libmxnet.so","libmxnet.dll"], ["$(ENV["MXNET_HOME"])/lib"])
#   if !isempty(lib)
#     info("Existing libmxnet detected at $lib")
#     exit(0)
#   else
#     info("Failed to load existing libmxnet, trying to build from source...")
#   end
# end


################################################################################
# If not found, try to build automatically using BinDeps
################################################################################
@windows_only begin
  info("Automatic building libxmnet on Windows is currently not supported.")
  info("Please follow the libmxnet documentation on how to build manually")
  info("or to install pre-build packages:")
  info("http://mxnet.readthedocs.org/en/latest/build.html#building-on-windows")
  exit(-1)
end

using BinDeps
@BinDeps.setup

#--------------------------------------------------------------------------------
# Install dependencies, opencv and blas
opencv_core = library_dependency("opencv_core", aliases=["libopencv_core"])

@linux_only begin
  provides(AptGet, "libopencv-dev", opencv_core)
  provides(Pacman, "opencv", opencv_core)
  provides(Yum, "opencv", opencv_core)

  blas = library_dependency("blas", aliases=["libblas","libblas.so.3"])
  provides(AptGet, "libblas-dev", blas)
  provides(Pacman, "blas", blas)
  provides(Yum, "blas-devel", blas)
end

@osx_only begin
  using Homebrew
  provides(Homebrew.HB, "opencv", opencv_core, os = :Darwin)

  # OSX has built-in BLAS we could use
end

@BinDeps.install Dict(:opencv_core => :opencv_core)
@linux_only begin
  @BinDeps.install Dict(:blas => :blas)
end

#--------------------------------------------------------------------------------
# Build libmxnet
mxnet = library_dependency("mxnet", aliases=["libmxnet"])

prefix = joinpath(BinDeps.depsdir(mxnet), "usr")
srcdir = joinpath(BinDeps.depsdir(mxnet),"src", "libmxnet")
libdir = joinpath(prefix, "lib")
provides(BuildProcess,
  (@build_steps begin
    CreateDirectory(srcdir)
    CreateDirectory(libdir)
    @build_steps begin
      ChangeDirectory(srcdir)
      `git clone --recursive https://github.com/dmlc/mxnet`
      FileRule(joinpath(libdir, "libmxnet.so"), @build_steps begin
        ChangeDirectory("mxnet")
        @osx_only `cp make/osx.mk config.mk`
        @osx_only `echo hahahahahahahaha=================`
        `make`
        `cp lib/libmxnet.so $libdir`
      end)
    end
  end), mxnet)

  @BinDeps.install Dict(:mxnet => :mxnet)
