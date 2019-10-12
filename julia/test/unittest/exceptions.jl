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

module TestExceptions

using MXNet
using Test

struct MXError′ <: AbstractMXError
  msg::String
end

function test_show()
  @info "AbstractMXError::Base.show"

  io = IOBuffer()
  e = MXError′("magic")
  print(io, e)
  str = String(take!(io))
  @test str == "magic"
end

@testset "Exception Test" begin
  test_show()
end


end  # module TestExceptions
