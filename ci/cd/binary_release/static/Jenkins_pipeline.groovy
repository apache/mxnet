// -*- mode: groovy -*-

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
//
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// utils = load('ci/Jenkinsfile_utils.groovy')

// libmxnet location
libmxnet = 'lib/libmxnet.so'

// licenses
licenses = 'licenses/*'

// libmxnet dependencies
mx_deps = 'lib/libgfortran.so.3, lib/libquadmath.so.0'
mx_mkldnn_deps = 'lib/libgfortran.so.3, lib/libquadmath.so.0, lib/libiomp5.so, lib/libmkldnn.so.0, lib/libmklml_intel.so, 3rdparty/mkldnn/build/install/include/mkldnn_version.h'

// settings
workspace_name = 'static_binary'
is_dynamic_binary = false

binary_release = load('ci/cd/binary_release/Jenkinsfile_binary_release.groovy')

// Builds the static binary for the specified mxnet variant
def build(mxnet_variant) {
  node('restricted-mxnetlinux-cpu') {
    ws("workspace/${workspace_name}/${mxnet_variant}/${env.BUILD_NUMBER}") {
      utils.init_git(env.MXNET_BRANCH, env.MXNET_SHA)
      utils.docker_run('publish.ubuntu1404_cpu', "build_static_python ${mxnet_variant}", false)
      utils.pack_lib("mxnet_${mxnet_variant}", binary_release.get_stash(mxnet_variant))
    }
  }
}

def get_pipeline(mxnet_variant) {
  return binary_release.get_pipeline(mxnet_variant, this.&build)
}

return this
