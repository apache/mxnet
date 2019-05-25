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

// To avoid confusion, please note:
// ci_utils and cd_utils are loaded by the originating Jenkins job, e.g. jenkins/Jenkinsfile_release_job

def get_pipeline(mxnet_variant, build_fn) {
  return {
    stage("${mxnet_variant}") {
      stage('Build') {
        timeout(time: max_time, unit: 'MINUTES') {
          build_fn(mxnet_variant)
        }
      }

      stage('Test') {
        def tests = [:]
        tests["${mxnet_variant}: Python 2"] = {
          stage("${mxnet_variant}: Python 2") {
            timeout(time: max_time, unit: 'MINUTES') {
              unittest_py2(mxnet_variant)
            }
          }
        }
        tests["${mxnet_variant}: Python 3"] = {
          stage("${mxnet_variant}: Python 3") {
            timeout(time: max_time, unit: 'MINUTES') {
              unittest_py3(mxnet_variant)
            }
          }
        }

        // Add quantization tests for all cu variants except cu80
        if (mxnet_variant.startsWith('cu') && !mxnet_variant.startsWith('cu80')) {
          tests["${mxnet_variant}: Quantization Python 3"] = {
            stage("${mxnet_variant}: Quantization Python 3") {
              timeout(time: max_time, unit: 'MINUTES') {
                test_gpu_quantization_py3(mxnet_variant)
              }
            }
          }
          tests["${mxnet_variant}: Quantization Python 2"] = {
            stage("${mxnet_variant}: Python 2") {
              timeout(time: max_time, unit: 'MINUTES') {
                test_gpu_quantization_py2(mxnet_variant)
              }
            }
          }
        }
        
        parallel tests
      }

      stage('Push') {
        timeout(time: max_time, unit: 'MINUTES') {
          push(mxnet_variant)
        }
      }
    }
  }
}

// Returns a string of comma separated resources to be stashed b/w stages
// E.g. the libmxnet library and any other dependencies
def get_stash(mxnet_variant) {
  def deps = mxnet_variant.endsWith('mkl') ? mx_mkldnn_deps : mx_deps
  return "${libmxnet}, ${licenses}, ${deps}"
}

// Returns the (Docker) environment for the given variant
// The environment corresponds to the docker files in the 'docker' directory
def get_environment(mxnet_variant) {
  if (mxnet_variant.startsWith("cu")) {
    // Remove 'mkl' suffix from variant to properly format test environment
    return "ubuntu_gpu_${mxnet_variant.replace('mkl', '')}"
  }
  return "ubuntu_cpu"
}

// Returns the variant appropriate jenkins node test in which
// to run a step
def get_jenkins_node_label(mxnet_variant) {
  if (mxnet_variant.startsWith('cu')) {
    return NODE_LINUX_GPU
  }
  return NODE_LINUX_CPU
}

// Runs unit tests using python 3
def unittest_py3(mxnet_variant) {
  def node_label = get_jenkins_node_label(mxnet_variant)

  node(node_label) {
    ws("workspace/mxnet_${libtype}/${mxnet_variant}/${env.BUILD_NUMBER}") {
      def image = get_environment(mxnet_variant)
      def use_nvidia_docker = mxnet_variant.startsWith('cu')
      ci_utils.unpack_and_init("mxnet_${mxnet_variant}", get_stash(mxnet_variant), false)
      ci_utils.docker_run(image, "cd_unittest_ubuntu ${mxnet_variant} python3", use_nvidia_docker)
    }
  }
}

// Runs unit tests using python 2
def unittest_py2(mxnet_variant) {
  def node_label = get_jenkins_node_label(mxnet_variant)
  node(node_label) {
    ws("workspace/mxnet_${libtype}/${mxnet_variant}/${env.BUILD_NUMBER}") {
      def image = get_environment(mxnet_variant)
      def use_nvidia_docker = mxnet_variant.startsWith('cu')
      ci_utils.unpack_and_init("mxnet_${mxnet_variant}", get_stash(mxnet_variant), false)
      ci_utils.docker_run(image, "cd_unittest_ubuntu ${mxnet_variant} python", use_nvidia_docker)
    }
  }
}

// Tests quantization in P3 instance using Python 2
def test_gpu_quantization_py2(mxnet_variant) {
  node(NODE_LINUX_GPU_P3) {
    ws("workspace/mxnet_${libtype}/${mxnet_variant}/${env.BUILD_NUMBER}") {
      def image = get_environment(mxnet_variant)
      ci_utils.unpack_and_init("mxnet_${mxnet_variant}", get_stash(mxnet_variant), false)
      ci_utils.docker_run(image, "unittest_ubuntu_python2_quantization_gpu", true)
    }
  }
}

// Tests quantization in P3 instance using Python 3
def test_gpu_quantization_py3(mxnet_variant) {
  node(NODE_LINUX_GPU_P3) {
    ws("workspace/mxnet_${libtype}/${mxnet_variant}/${env.BUILD_NUMBER}") {
      def image = get_environment(mxnet_variant)
      ci_utils.unpack_and_init("mxnet_${mxnet_variant}", get_stash(mxnet_variant), false)
      ci_utils.docker_run(image, "unittest_ubuntu_python3_quantization_gpu", true)
    }
  }
}

// Pushes artifact to artifact repository
def push(mxnet_variant) {
  node(NODE_LINUX_CPU) {
    ws("workspace/mxnet_${libtype}/${mxnet_variant}/${env.BUILD_NUMBER}") {
      def deps = (mxnet_variant.endsWith('mkl')? mx_mkldnn_deps : mx_deps).replaceAll(',', '')
      ci_utils.unpack_and_init("mxnet_${mxnet_variant}", get_stash(mxnet_variant), false)
      cd_utils.push_artifact(libmxnet, mxnet_variant, libtype, licenses, deps)
    }
  }
}

return this
