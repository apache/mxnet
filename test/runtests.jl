using MXNet
using Base.Test

# run test in the whole directory, latest modified files
# are run first, this makes waiting time shorter when writing
# or modifying unit-tests
function test_dir(dir)
  jl_files = sort(filter(x -> ismatch(r".*\.jl$", x), readdir(dir)), by = fn -> stat(joinpath(dir,fn)).mtime)
  map(reverse(jl_files)) do file
    include("$dir/$file")
  end
end

include("common.jl")
test_dir("unittest")

# run the basic MNIST mlp example
if haskey(ENV, "CONTINUOUS_INTEGRATION")
  include(joinpath(Pkg.dir("MXNet"), "examples", "mnist", "mlp-test.jl"))
end
