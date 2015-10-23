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
# Install dependencies, blas
@linux_only begin
  blas = library_dependency("blas", aliases=["libblas","libblas.so.3"])
  provides(AptGet, "libblas-dev", blas)
  provides(Pacman, "blas", blas)
  provides(Yum, "blas-devel", blas)

  @BinDeps.install Dict(:blas => :blas)
end

#--------------------------------------------------------------------------------
# Build libmxnet
mxnet = library_dependency("mxnet", aliases=["libmxnet.so"])

prefix = joinpath(BinDeps.depsdir(mxnet), "usr")
srcdir = joinpath(BinDeps.depsdir(mxnet),"src")
mxdir  = joinpath(srcdir, "mxnet")
libdir = joinpath(prefix, "lib")
provides(BuildProcess,
  (@build_steps begin
    CreateDirectory(srcdir)
    CreateDirectory(libdir)
    @build_steps begin
      ChangeDirectory(srcdir)
      `rm -rf mxnet`
      `git clone --recursive https://github.com/dmlc/mxnet`
      FileRule(joinpath(libdir, "libmxnet.so"), @build_steps begin
        ChangeDirectory("$mxdir")
        `cp make/config.mk config.mk`
        @osx_only `cp make/osx.mk config.mk`
        `sed -i -s 's/USE_OPENCV = 1/USE_OPENCV = 0/' config.mk`
        `make`
        `cp lib/libmxnet.so $libdir`
      end)
    end
  end), mxnet)

  @BinDeps.install Dict(:mxnet => :mxnet)
