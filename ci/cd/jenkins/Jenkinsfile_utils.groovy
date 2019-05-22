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

// returns the value of the MXNET_SHA env. variable, if set
// or the commit id at the head of the branch being built
def get_mxnet_commit_id_for_build() {
  return sh(returnStdout: true, script: '''git ls-remote https://github.com/apache/incubator-mxnet.git ${MXNET_BRANCH} | awk '{ print $1 }' ''').trim()
}

def trigger_release_job(job_name, job_type, mxnet_variants) {
  def run = build(
    job: 'restricted-mxnet-cd/mxnet-cd-release-job', 
    parameters: [
      string(name: 'MXNET_BRANCH', value: "${env.MXNET_BRANCH}"),
      string(name: 'MXNET_SHA', value: "${env.MXNET_SHA}"),
      string(name: 'RELEASE_JOB_NAME', value: "${job_name}"),
      string(name: 'RELEASE_JOB_TYPE', value: "${job_type}"),
      string(name: 'RELEASE_JOB_TYPE', value: "${job_type}"),
      string(name: 'MXNET_VARIANTS', value: "${mxnet_variants}"),
      booleanParam(name: 'RELEASE_BUILD', value: "${env.RELEASE_BUILD}")
    ], 
    propagate: false)

  def result = run.getResult()

  if (result == "UNSTABLE") {
    currentBuild.result = "UNSTABLE" 
  }

  if (result == "FAILURE") {
    error "Downstream job: ${job_name} failed"
  }

  if (result == "ABORTED") {
    error "Downstream job: ${job_name} was aborted" 
  }
}

// A generic pipeline that can be used by *most* CD jobs
// It can use used when implementing the pipeline steps in the Jenkins_steps.groovy
// script for the particular delivery channel. However, it should also implemente the
// build, test, and push steps.
// NOTE: Be mindful of the expected time that a step should take. If it will take a long time,
// and it can be done in a CPU node, do it in a CPU node. We should avoid using GPU instances unless
// we *have* to.
// However, if it is only packaging libmxnet and that doesn't take long. The the pipeline can 
// just run on a single node. As is done bellow.
// For examples of multi-node CD pipelines, see the the binary_release/static and binary_release/dynamic
// pipeline.
def release_pipeline(mxnet_variant, custom_steps, node_type = "restricted-mxnetlinux-cpu") {
  return ["${mxnet_variant}": {
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
  }]
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

return this
