<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Workflow for making a release

1. Update `NEWS.md` to list important changes
2. Check out the `stable` branch, merge with `master`.
3. Update `libmxnet_curr_ver` in `deps/build.jl` to the latest commit SHA (or any proper reference). Using `master` here is not good because future changes in libmxnet might break existing Julia packages.
4. Run tests.
5. Merge master into stable branch.
6. Tag stable branch: `git tag v1.2.3`
7. Push tag to remote: `git push origin <tagname>`
8. Edit the [releases page](https://github.com/dmlc/MXNet.jl/releases)
   to copy the release notes from `NEWS.md` to the newly created release tag.
9. Goto https://github.com/JuliaLang/METADATA.jl/pulls
   and check `attobot` already make a PR for the release.
