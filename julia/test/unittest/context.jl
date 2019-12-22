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

module TestContext

using MXNet
using Test

function test_num_gpus()
  @info "Context::num_gpus"

  @test num_gpus() >= 0
end

function test_context_macro()
  @info "Context::@context"

  @context mx.CPU 42 begin
    ctx = mx.current_context()
    @test ctx.device_type == mx.CPU
    @test ctx.device_id   == 42

    @context mx.GPU 24 begin
      ctx = mx.current_context()
      @test ctx.device_type == mx.GPU
      @test ctx.device_id   == 24
    end

    ctx = mx.current_context()
    @test ctx.device_type == mx.CPU
    @test ctx.device_id   == 42
  end

  function f()
    ctx = mx.current_context()
    @test ctx.device_type == mx.GPU
    @test ctx.device_id   == 123
  end

  @context mx.GPU 123 begin
    f()
  end

  @context mx.GPU begin
    ctx = mx.current_context()
    @test ctx.device_type == mx.GPU
    @test ctx.device_id   == 0
  end

  @context mx.CPU begin
    ctx = mx.current_context()
    @test ctx.device_type == mx.CPU
    @test ctx.device_id   == 0
  end

  @info "Context::@gpu"
  @gpu 123 f()
  @gpu begin
    ctx = mx.current_context()
    @test ctx.device_type == mx.GPU
    @test ctx.device_id   == 0
  end
  let n = 321
    @gpu n begin
      ctx = mx.current_context()
      @test ctx.device_type == mx.GPU
      @test ctx.device_id   == 321
    end
  end

  @info "Context::@cpu"
  @cpu 123 begin
    ctx = mx.current_context()
    @test ctx.device_type == mx.CPU
    @test ctx.device_id   == 123
  end
  @cpu begin
    ctx = mx.current_context()
    @test ctx.device_type == mx.CPU
    @test ctx.device_id   == 0
  end
  let n = 321
    @cpu n begin
      ctx = mx.current_context()
      @test ctx.device_type == mx.CPU
      @test ctx.device_id   == 321
    end
  end
end

@testset "Context Test" begin
  test_num_gpus()
  test_context_macro()
end


end  # module TestContext
