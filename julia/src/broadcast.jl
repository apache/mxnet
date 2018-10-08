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

using TakingBroadcastSeriously: Broadcasted, unwrap

for f in :[%,
           tan, asin, acos, atan,
           sinh, cosh, tanh, asinh, acosh, atanh,
           min, max,
           hypot].args
  # copy from TakingBroadcastSeriously
  @eval Base.$f(a::Broadcasted...) = Broadcasted(broadcast_($f, unwrap.(a)...))
  @eval Base.$f(a::Broadcasted, b) = Broadcasted(broadcast_($f, unwrap(a), b))
  @eval Base.$f(b, a::Broadcasted) = Broadcasted(broadcast_($f, b, unwrap(a)))
end

for f in :[σ, sigmoid, relu, softmax, log_softmax].args
  # copy from TakingBroadcastSeriously
  @eval $f(a::Broadcasted...) = Broadcasted(broadcast_($f, unwrap.(a)...))
  @eval $f(a::Broadcasted, b) = Broadcasted(broadcast_($f, unwrap(a), b))
  @eval $f(b, a::Broadcasted) = Broadcasted(broadcast_($f, b, unwrap(a)))
end
