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

module TestNameManager
using MXNet
using Base.Test

function test_default()
  info("NameManager::default")

  name = :_____aaaaa_____
  @test get!(mx.DEFAULT_NAME_MANAGER, name, "") == name
  @test get!(mx.DEFAULT_NAME_MANAGER, string(name), "") == name

  hint = name
  @test get!(mx.DEFAULT_NAME_MANAGER, "", hint) == Symbol("$(hint)0")
  @test get!(mx.DEFAULT_NAME_MANAGER, "", string(hint)) == Symbol("$(hint)1")
end

function test_prefix()
  info("NameManager::prefix")

  name   = :_____bbbbb_____
  prefix = :_____foobar_____

  prefix_manager = mx.PrefixNameManager(prefix)
  @test get!(prefix_manager, name, "") == Symbol("$prefix$name")
  @test get!(prefix_manager, "", name) == Symbol("$prefix$(name)0")
end

@testset "Name Test" begin
  test_default()
  test_prefix()
end

end
