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

// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// mxnet libraries
mx_lib = 'lib/libmxnet.so, lib/libmxnet.a, 3rdparty/dmlc-core/libdmlc.a, 3rdparty/tvm/nnvm/lib/libnnvm.a'
// mxnet cmake libraries, in cmake builds we do not produce a libnvvm static library by default.
mx_cmake_lib = 'build/libmxnet.so, build/libmxnet.a, build/3rdparty/dmlc-core/libdmlc.a, build/tests/mxnet_unit_tests, build/3rdparty/openmp/runtime/src/libomp.so'
// timeout in minutes
max_time = 120
// assign any caught errors here
err = null

// initialize source codes
def init_git() {
  deleteDir()
  retry(5) {
    try {
      // Make sure wait long enough for api.github.com request quota. Important: Don't increase the amount of
      // retries as this will increase the amount of requests and worsen the throttling
      timeout(time: 15, unit: 'MINUTES') {
        checkout scm
        sh 'git submodule update --init --recursive'
        sh 'git clean -d -f'
      }
    } catch (exc) {
      deleteDir()
      error "Failed to fetch source codes with ${exc}"
      sleep 2
    }
  }
}

// pack libraries for later use
def pack_lib(name, libs=mx_lib) {
  sh """
echo "Packing ${libs} into ${name}"
echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
"""
  stash includes: libs, name: name
}

// unpack libraries saved before
def unpack_lib(name, libs=mx_lib) {
  unstash name
  sh """
echo "Unpacked ${libs} from ${name}"
echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
"""
}

def publish_test_coverage() {
    // Fall back to our own copy of the bash helper if it failed to download the public version
    sh '(curl --retry 10 -s https://codecov.io/bash | bash -s -) || (curl --retry 10 -s https://s3-us-west-2.amazonaws.com/mxnet-ci-prod-slave-data/codecov-bash.txt | bash -s -)'
}

def collect_test_results_unix(original_file_name, new_file_name) {
    if (fileExists(original_file_name)) {
        // Rename file to make it distinguishable. Unfortunately, it's not possible to get STAGE_NAME in a parallel stage
        // Thus, we have to pick a name manually and rename the files so that they can be stored separately.
        sh 'cp ' + original_file_name + ' ' + new_file_name
        archiveArtifacts artifacts: new_file_name
    }
}

def collect_test_results_windows(original_file_name, new_file_name) {
    // Rename file to make it distinguishable. Unfortunately, it's not possible to get STAGE_NAME in a parallel stage
    // Thus, we have to pick a name manually and rename the files so that they can be stored separately.
    if (fileExists(original_file_name)) {
        bat 'xcopy ' + original_file_name + ' ' + new_file_name + '*'
        archiveArtifacts artifacts: new_file_name
    }
}

def docker_run(platform, function_name, use_nvidia, shared_mem = '500m') {
  def command = "ci/build.py --docker-registry ${env.DOCKER_CACHE_REGISTRY} %USE_NVIDIA% --platform %PLATFORM% --docker-build-retries 3 --shm-size %SHARED_MEM% /work/runtime_functions.sh %FUNCTION_NAME%"
  command = command.replaceAll('%USE_NVIDIA%', use_nvidia ? '--nvidiadocker' : '')
  command = command.replaceAll('%PLATFORM%', platform)
  command = command.replaceAll('%FUNCTION_NAME%', function_name)
  command = command.replaceAll('%SHARED_MEM%', shared_mem)

  sh command
}

def python3_gpu_ut(docker_container_name) {
  timeout(time: max_time, unit: 'MINUTES') {
    docker_run(docker_container_name, 'unittest_ubuntu_python3_gpu', true)
  }
}


try {
  stage('Sanity Check') {
    parallel 'Lint': {
      node('mxnetlinux-cpu') {
        ws('workspace/sanity-lint') {
          init_git()
          docker_run('ubuntu_cpu', 'sanity_check', false)
        }
      }
    },
    'RAT License': {
      node('mxnetlinux-cpu') {
        ws('workspace/sanity-rat') {
          init_git()
          docker_run('ubuntu_rat', 'nightly_test_rat_check', false)
        }
      }
    }
  }

  stage('Build') {
    'GPU: Make': {
      node('mxnetlinux-cpu') {
        ws('workspace/build-gpu-make') {
          timeout(time: max_time, unit: 'MINUTES') {
            init_git()
            docker_run('ubuntu_gpu', 'build_ubuntu_gpu_make', false)
            pack_lib('make_gpu', mx_lib)
          }
        }
      }
    },
    'GPU: CMake': {
      node('mxnetlinux-cpu') {
        ws('workspace/build-gpu-cmake') {
          timeout(time: max_time, unit: 'MINUTES') {
            init_git()
            docker_run('ubuntu_gpu', 'build_ubuntu_gpu_cmake', false)
            pack_lib('cmake_gpu', mx_cmake_lib)
          }
        }
      }
    }
  } // End of stage('Build')

  stage('Tests') {
    'Python3: GPU, Make': {
      node('mxnetlinux-gpu') {
        ws('workspace/ut-python3-gpu-make') {
          try {
            init_git()
            unpack_lib('make_gpu', mx_lib)
            python3_gpu_ut('ubuntu_gpu')
            publish_test_coverage()
          } finally {
            collect_test_results_unix('nosetests_gpu.xml', 'nosetests_python3_gpu.xml')
          }
        }
      }
    },
    'Python3: GPU, CMake': {
      node('mxnetlinux-gpu') {
        ws('workspace/ut-python3-gpu-cmake') {
          try {
            init_git()
            unpack_lib('cmake_gpu', mx_cmake_lib)
            python3_gpu_ut('ubuntu_gpu')
            publish_test_coverage()
          } finally {
            collect_test_results_unix('nosetests_gpu.xml', 'nosetests_python3_gpu.xml')
          }
        }
      }
    }
  }

  stage('Deploy') {
    node('mxnetlinux-cpu') {
      ws('workspace/docs') {
        timeout(time: max_time, unit: 'MINUTES') {
          init_git()
          docker_run('ubuntu_cpu', 'deploy_docs', false)
          sh "tests/ci_build/deploy/ci_deploy_doc.sh ${env.BRANCH_NAME} ${env.BUILD_NUMBER}"
        }
      }
    }
  }

  // set build status to success at the end
  currentBuild.result = "SUCCESS"
} catch (caughtError) {
  node("mxnetlinux-cpu") {
    sh "echo caught ${caughtError}"
    err = caughtError
    currentBuild.result = "FAILURE"
  }
} finally {
  node("mxnetlinux-cpu") {
    // Only send email if master or release branches failed
    if (currentBuild.result == "FAILURE" && (env.BRANCH_NAME == "master" || env.BRANCH_NAME.startsWith("v"))) {
      emailext body: 'Build for MXNet branch ${BRANCH_NAME} has broken. Please view the build at ${BUILD_URL}', replyTo: '${EMAIL}', subject: '[BUILD FAILED] Branch ${BRANCH_NAME} build ${BUILD_NUMBER}', to: '${EMAIL}'
    }
    // Remember to rethrow so the build is marked as failing
    if (err) {
      throw err
    }
  }
}
