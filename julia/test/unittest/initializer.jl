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

@testset "Initializers" begin
  @testset "Bilinear initializer" begin
    # Setup a filter with scale = 2
    expectedFilter = Float32[
                       0.0625 0.1875 0.1875 0.0625;
                       0.1875 0.5625 0.5625 0.1875;
                       0.1875 0.5625 0.5625 0.1875;
                       0.0625 0.1875 0.1875 0.0625]
    filter = mx.zeros(Float32, 4, 4, 1, 4)
    mx.init(mx.XavierInitializer(), :upsampling0_weight, filter)

    mx.@nd_as_jl ro=filter begin
      for s in 1:size(filter, 4)
        @test all(filter[:, :, 1, s] .== expectedFilter)
      end
    end
  end
end
