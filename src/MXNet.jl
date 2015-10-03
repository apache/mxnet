module MXNet

# we put everything in the namespace mx, because there are a lot of
# functions with the same names as built-in utilities like "zeros", etc.
export mx
module mx

include("init.jl")
include("context.jl")
include("ndarray.jl")

end # mx

end # module MXNet
