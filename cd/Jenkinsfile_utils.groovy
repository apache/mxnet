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

// Triggers a downstream jenkins job responsible for building, testing
// and publishing all the variants for a particular 'job_type'.
// The 'job_type' should be the name of the directory that contains the 
// 'Jenkins_pipeline.groovy' file and has the pipeline definition for the 
// artifact (docker image, binary, pypi or maven package, etc.) that should
// be published.

STATE_UPDATE="State Update"

def trigger_release_job(job_name, job_type, mxnet_variants) {
  def run = build(
    job: env.CD_RELEASE_JOB_NAME, 
    parameters: [
      string(name: "RELEASE_JOB_NAME", value: "${job_name}"),
      string(name: "RELEASE_JOB_TYPE", value: "${job_type}"),
      string(name: "MXNET_VARIANTS", value: "${mxnet_variants}"),
      booleanParam(name: "RELEASE_BUILD", value: "${env.RELEASE_BUILD}"),
      string(name: "COMMIT_ID", value: "${env.GIT_COMMIT}")
    ],
    // If propagate is true, any result other than successful will
    // mark this call as failure (inc. unstable).
    // https://jenkins.io/doc/pipeline/steps/pipeline-build-step
    propagate: false)

  def result = run.getResult()

  // In case the underlying release job is unstable,
  // e.g. one or more (but not all) the variants failed, or;
  // it is aborted (e.g. one of steps timed out),
  // continue with the pipeline and try to post as many releases as possible
  // but mark it as unstable
  if (result == "UNSTABLE" || result == "ABORTED") {
    currentBuild.result = "UNSTABLE" 
  }

  // Throw an exception on failure, because this would mean the whole
  // pipeline failed (i.e. for every variant)
  if (result == "FAILURE") {
    error "Downstream job: ${job_name} failed"
  }
}


// This triggers a downstream release job with no
// variants and not job type. This will update
// the configuration of the release job in jenkins
// to the configuration of release job as defined in the
// Jenkinsfile _release_job for env.GIT_COMMIT revision
def update_release_job_state() {
  build(
    job: env.CD_RELEASE_JOB_NAME, 
    parameters: [
      string(name: "RELEASE_JOB_TYPE", value: STATE_UPDATE),
  
      // Should be set to the current git commit
      string(name: "COMMIT_ID", value: "${env.GIT_COMMIT}")
    ])
}

// Wraps variant pipeline with error catching and
// job status setting code
// If there's an error in one of the pipelines, set status to UNSTABLE
// If all pipelines fail, set to FAILURE
// This is to be used in conjunction with the error_checked_parallel
def wrap_variant_pipeline_fn(variant_pipeline, total_num_pipelines) {
  // do not add def - seems to affect the scope
  count = 0
  return {
    try {
      variant_pipeline()
    } catch (ex) {
      count++
      currentBuild.result = "UNSTABLE"

      if (count == total_num_pipelines) {
        currentBuild.result = "FAILURE"
        throw ex
      }
    }
  }
}

// Takes a map of key -> closure values to be executed in parallel.
// The outcome of the execution of each parallel step will affect
// the result (SUCCESS, FAILURE, ABORTED, UNSTABLE) of the overall job.
// If all steps fail or are aborted, the job will be set to failed.
// If some steps fail or are aborted, the job will be set to unstable. 
def error_checked_parallel(variant_pipelines) {
  pipelines = variant_pipelines.inject([:]) { mp, key, value ->
    mp << ["${key}": wrap_variant_pipeline_fn(value, variant_pipelines.size())]
  }
  parallel pipelines
}

// pushes artifact to repository
def push_artifact(libmxnet_path, variant, libtype, license_paths = '', dependency_paths = '') {
  if(license_paths == null) license_paths = ''
  if(dependency_paths == null) dependency_paths = ''

  sh "./cd/utils/artifact_repository.py --push --verbose --libtype ${libtype} --variant ${variant} --libmxnet ${libmxnet_path} --licenses ${license_paths} --dependencies ${dependency_paths}"
}

// pull artifact from repository
def pull_artifact(variant, libtype, destination = '') {
  sh "./cd/utils/artifact_repository.py --pull --verbose --libtype ${libtype} --variant ${variant} --destination ${destination}"
}

// pulls artifact from repository and places files in the appropriate directories
def restore_artifact(variant, libtype) {

  pull_artifact(variant, libtype, 'mxnet_artifact')

  // move libraries to lib directory
  dir('lib') {
    sh "mv ../mxnet_artifact/libmxnet.so ."
    if (fileExists('../mxnet_artifact/dependencies')) {
      sh """find "../mxnet_artifact/dependencies" -type f -name "*.so*" -exec mv {} . \\;"""
      sh "ls ."
    }
  }

  dir('cd_misc') {
    if (fileExists('../mxnet_artifact/dependencies')) {
      // All library files (*.so*) should have be moved
      // to the lib directory. If anything is left, it will be
      // other supporting files (header files, etc.)
      sh """find "../mxnet_artifact/dependencies" -type f -exec mv {} . \\;"""
      sh "ls ."
    }
  }

  dir('licenses') {
    if (fileExists('../mxnet_artifact/licenses')) {
      sh """find "../mxnet_artifact/licenses" -type f -exec mv {} . \\;"""
      sh "ls ."
    }
  }

  dir('mxnet_artifact') {
    deleteDir()
  }
}


// Restores the statically linked libmxnet for the given variant
def restore_static_libmxnet(variant) {
  restore_artifact(variant, 'static')
}


// Restores the dynamically linked libmxnet for the given variant
def restore_dynamic_libmxnet(variant) {
  restore_artifact(variant, 'dynamic')
}

// A generic pipeline that can be used by *most* CD jobs
// It can be used when implementing the pipeline steps in the Jenkins_steps.groovy
// script for a particular delivery channel. However, it should also implement the
// build, test, and push steps.
// NOTE: Be mindful of the expected time that a step should take. If it will take a long time,
// and it can be done in a CPU node, do it in a CPU node. We should avoid using GPU instances unless
// we *have* to.
// However, if it is only packaging libmxnet and that doesn't take long. Then, the pipeline can 
// just run on a single node. As is done bellow.
// For examples of multi-node CD pipelines, see the the binary_release/static and binary_release/dynamic
// pipeline.
def generic_pipeline(mxnet_variant, custom_steps, node_type = "restricted-mxnetlinux-cpu") {
  return {
    node(node_type) {
      stage("${mxnet_variant}") {

        stage('Build') {
          custom_steps.build(mxnet_variant)
        }

        stage('Test') {
          custom_steps.test(mxnet_variant)
        }

        stage('Push') {
          custom_steps.push(mxnet_variant)
        }
      }
    }
  }
}

return this
