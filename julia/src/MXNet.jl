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

module MXNet

using Reexport

# we put everything in the namespace mx, because there are a lot of
# functions with the same names as built-in utilities like "zeros", etc.
export mx
module mx

using Base.Broadcast: Broadcasted, DefaultArrayStyle
using Libdl
using LinearAlgebra
using Markdown
using Printf
using Statistics
using Random

using Formatting
using MacroTools

# Functions from base that we can safely extend and that are defined by libmxnet.
import Base.Broadcast: broadcasted
import Base.Iterators: filter

###############################################################################
#  exports
###############################################################################

# exceptions.jl
export AbstractMXError,
       MXError

# symbolic-node.jl
export SymbolicNode,
       Variable,
       @var

# ndarray.jl
export NDArray,
       context,
       expand_dims,
       @inplace,
       # activation funcs
       σ,
       sigmoid,
       relu,
       softmax,
       log_softmax,
       # broadcast utils
       broadcast_to,
       broadcast_axis,
       broadcast_axes

# executor.jl
export Executor,
       bind,
       simple_bind,
       forward,
       backward

# context.jl
export Context,
       cpu,
       gpu,
       num_gpus,
       gpu_memory_info

# model.jl
export AbstractModel,
       FeedForward,
       predict

# nn-factory.jl
export MLP

# metric.jl
export AbstractEvalMetric,
       ACE,
       Accuracy,
       MSE,
       MultiACE,
       MultiMetric,
       NMSE,
       SeqMetric

# kvstore.jl
export KVStore,
       init!,
       pull!,
       barrier,
       setoptimizer!,
       setupdater!

# initializer.jl
export AbstractInitializer,
       UniformInitializer,
       NormalInitializer,
       XavierInitializer

# optimizer.jl
export AbstractOptimizer,
       AdaDelta,
       AdaGrad,
       ADAM,
       AdaMax,
       Nadam,
       RMSProp,
       SGD,
       getupdater,
       normgrad!,
       update!

# io.jl
export AbstractDataProvider,
       AbstractDataBatch,
       DataBatch,
       ArrayDataProvider,
       ArrayDataBatch

# visualize.jl
export to_graphviz

###############################################################################
#  includes
###############################################################################

include("exceptions.jl")
include("base.jl")

include("runtime.jl")
include("context.jl")
include("util.jl")

include("ndarray.jl")

include("random.jl")
include("autograd.jl")

include("name.jl")
include("symbolic-node.jl")
include("executor.jl")

include("broadcast.jl")

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

@reexport using .mx

end # module MXNet
