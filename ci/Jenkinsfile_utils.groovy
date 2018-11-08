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

// initialize source codes
def init_git() {
  deleteDir()
  retry(5) {
    try {
      // Make sure wait long enough for api.github.com request quota. Important: Don't increase the amount of
      // retries as this will increase the amount of requests and worsen the throttling
      timeout(time: 15, unit: 'MINUTES') {
        checkout scm
        sh 'git clean -xdff'
        sh 'git reset --hard'
        sh 'git submodule update --init --recursive'
        sh 'git submodule foreach --recursive git clean -ffxd'
        sh 'git submodule foreach --recursive git reset --hard'
      }
    } catch (exc) {
      deleteDir()
      error "Failed to fetch source codes with ${exc}"
      sleep 2
    }
  }
}

def init_git_win() {
  deleteDir()
  retry(5) {
    try {
      // Make sure wait long enough for api.github.com request quota. Important: Don't increase the amount of
      // retries as this will increase the amount of requests and worsen the throttling
      timeout(time: 15, unit: 'MINUTES') {
        checkout scm
        bat 'git clean -xdff'
        bat 'git reset --hard'
        bat 'git submodule update --init --recursive'
        bat 'git submodule foreach --recursive git clean -ffxd'
        bat 'git submodule foreach --recursive git reset --hard'
      }
    } catch (exc) {
      deleteDir()
      error "Failed to fetch source codes with ${exc}"
      sleep 2
    }
  }
}

// pack libraries for later use
def pack_lib(name, libs, include_gcov_data = false) {
  sh """
echo "Packing ${libs} into ${name}"
echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
"""
  stash includes: libs, name: name

  if (include_gcov_data) {
    // Store GCNO files that are required for GCOV to operate during runtime
    sh "find . -name '*.gcno'"
    stash name: "${name}_gcov_data", includes: "**/*.gcno"
  }
}

// unpack libraries saved before
def unpack_and_init(name, libs, include_gcov_data = false) {
  init_git()
  unstash name
  sh """
echo "Unpacked ${libs} from ${name}"
echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
"""
  if (include_gcov_data) {
    // Restore GCNO files that are required for GCOV to operate during runtime
    unstash "${name}_gcov_data"
  }
}

def publish_test_coverage() {
    // CodeCovs auto detection has trouble with our CIs PR validation due the merging strategy
    lastCommitMessage = sh (script: "git log -1 --pretty=%B", returnStdout: true)
    lastCommitMessage = lastCommitMessage.trim()
    if (lastCommitMessage.startsWith("Merge commit '") && lastCommitMessage.endsWith("' into HEAD")) {
        // Merge commit applied by Jenkins, skip that commit
        GIT_COMMIT_HASH = sh (script: "git rev-parse @~", returnStdout: true)
    } else {
        GIT_COMMIT_HASH = sh (script: "git rev-parse @", returnStdout: true)
    }
   
    if (env.CHANGE_ID) {
      // PR execution
      codecovArgs = "-B ${env.CHANGE_TARGET} -C ${GIT_COMMIT_HASH} -P ${env.CHANGE_ID}"
    } else {
      // Branch execution
      codecovArgs = "-B ${env.BRANCH_NAME} -C ${GIT_COMMIT_HASH}"
    }

    // To make sure we never fail because test coverage reporting is not available
    // Fall back to our own copy of the bash helper if it failed to download the public version
    sh "(curl --retry 10 -s https://codecov.io/bash | bash -s - ${codecovArgs}) || (curl --retry 10 -s https://s3-us-west-2.amazonaws.com/mxnet-ci-prod-slave-data/codecov-bash.txt | bash -s - ${codecovArgs}) || true"
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



def assign_node_labels(args) {
  NODE_LINUX_CPU = args.linux_cpu
  NODE_LINUX_GPU = args.linux_gpu
  NODE_LINUX_GPU_P3 = args.linux_gpu_p3
  NODE_WINDOWS_CPU = args.windows_cpu
  NODE_WINDOWS_GPU = args.windows_gpu
}

def main_wrapper(args) {
  // Main Jenkinsfile pipeline wrapper handler that allows to wrap core logic into a format
  // that supports proper failure handling
  // args:
  // - core_logic: Jenkins pipeline containing core execution logic
  // - failure_handler: Failure handler
  
  // assign any caught errors here
  err = null
  try {
    args['core_logic']()

    // set build status to success at the end
    currentBuild.result = "SUCCESS"
  } catch (caughtError) {
    node(NODE_LINUX_CPU) {
      sh "echo caught ${caughtError}"
      err = caughtError
      currentBuild.result = "FAILURE"
    }
  } finally {
    node(NODE_LINUX_CPU) {
      // Call failure handler
      args['failure_handler']()
      
      // Remember to rethrow so the build is marked as failing
      if (err) {
        throw err
      }
    }
  }
}
return this
