__precompile__()

module MXNet

# we put everything in the namespace mx, because there are a lot of
# functions with the same names as built-in utilities like "zeros", etc.
export mx
module mx

using Compat
import Compat.String
import Compat.view

import Base.Iterators: filter

using Formatting
using MacroTools
using TakingBroadcastSeriously: @unfuse
import TakingBroadcastSeriously: broadcast_

# Functions from base that we can safely extend and that are defined by libmxnet.
import Base: round, ceil, floor, cos, sin, abs, sign, exp, sqrt, exp, log, norm,
             transpose

include("base.jl")

include("compat.jl")

include("context.jl")
include("util.jl")

include("ndarray.jl")
include("random.jl")

include("name.jl")
include("symbolic-node.jl")
include("executor.jl")

include("metric.jl")
include("optimizer.jl")
include("initializer.jl")

include("io.jl")
include("kvstore.jl")

include("callback.jl")
include("model.jl")

include("visualize.jl")

include("nn-factory.jl")

include("deprecated.jl")

end # mx

end # module MXNet
