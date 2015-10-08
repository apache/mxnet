module TestNameManager
using MXNet
using Base.Test

function test_default()
  info("NameManager::default")

  name = :_____aaaaa_____
  @test get!(mx.DEFAULT_NAME_MANAGER, name, "") == name
  @test get!(mx.DEFAULT_NAME_MANAGER, string(name), "") == name

  hint = name
  @test get!(mx.DEFAULT_NAME_MANAGER, "", hint) == symbol("$(hint)0")
  @test get!(mx.DEFAULT_NAME_MANAGER, "", string(hint)) == symbol("$(hint)1")
end

function test_prefix()
  info("NameManager::prefix")

  name   = :_____bbbbb_____
  prefix = :_____foobar_____

  prefix_manager = mx.PrefixNameManager(prefix)
  @test get!(prefix_manager, name, "") == symbol("$prefix$name")
  @test get!(prefix_manager, "", name) == symbol("$prefix$(name)0")
end

test_default()
test_prefix()

end
