// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// mxnet libraries
mx_lib = 'lib/libmxnet.so, lib/libmxnet.a, dmlc-core/libdmlc.a, nnvm/lib/libnnvm.a'

//cpp test executable
cpp_test = 'build/tests/mxnet_test'

// command to start a docker container
docker_run = 'tests/ci_build/ci_build.sh'
// timeout in minutes
max_time = 120
// assign any caught errors here
err = null

// initialize source codes
def init_git() {
  retry(5) {
    try {
      timeout(time: 2, unit: 'MINUTES') {
        checkout scm
        sh 'git submodule update --init'
        sh 'git clean -d -f'        
      }
    } catch (exc) {
      deleteDir()
      error "Failed to fetch source codes"
      sleep 2
    }
  }
}

def init_git_win() {
  retry(5) {
    try {
      timeout(time: 2, unit: 'MINUTES') {
        checkout scm
        bat 'git submodule update --init'
        bat 'git clean -d -f'        
      }
    } catch (exc) {
      deleteDir()
      error "Failed to fetch source codes"
      sleep 2
    }
  }
}

// Run make. First try to do an incremental make from a previous workspace in hope to
// accelerate the compilation. If something wrong, clean the workspace and then
// build from scratch.
def make(docker_type, make_flag) {
  timeout(time: max_time, unit: 'MINUTES') {
    try {
      sh "${docker_run} ${docker_type} make ${make_flag}"
    } catch (exc) {
      echo 'Incremental compilation failed. Fall back to build from scratch'
      sh "${docker_run} ${docker_type} sudo make clean"
      sh "${docker_run} ${docker_type} sudo make -C amalgamation/ clean"
      sh "${docker_run} ${docker_type} make ${make_flag}"
    }
  }
}

// Run make. First try to do an incremental make from a previous workspace in hope to
// accelerate the compilation. If something wrong, clean the workspace and then
// build from scratch.
def make_test(docker_type, make_flag) {
  timeout(time: max_time, unit: 'MINUTES') {
    try {
      sh "${docker_run} ${docker_type} make test ${make_flag}"
    } catch (exc) {
      echo 'Incremental compilation failed. Fall back to build from scratch'
      sh "${docker_run} ${docker_type} rm -rf build/tests/"
      sh "${docker_run} ${docker_type} make test ${make_flag}"
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

def cpp_ut(docker_type) {
  timeout(time: max_time, unit: 'MINUTES') {
    sh "${docker_run} ${docker_type} ./build/tests/cpp/mxnet_test"
  }
}

// Python unittest for CPU
// Python 2
def python2_ut(docker_type) {
  timeout(time: max_time, unit: 'MINUTES') {
    sh "${docker_run} ${docker_type} find . -name '*.pyc' -type f -delete"
    sh "${docker_run} ${docker_type} PYTHONPATH=./python/ nosetests --with-timer --verbose tests/python/unittest"
    sh "${docker_run} ${docker_type} PYTHONPATH=./python/ nosetests --with-timer --verbose tests/python/train"
  }
}

// Python 3
def python3_ut(docker_type) {
  timeout(time: max_time, unit: 'MINUTES') {
    sh "${docker_run} ${docker_type} find . -name '*.pyc' -type f -delete"
    sh "${docker_run} ${docker_type} PYTHONPATH=./python/ nosetests-3.4 --with-timer --verbose tests/python/unittest"
  }
}

// GPU test has two parts. 1) run unittest on GPU, 2) compare the results on
// both CPU and GPU
// Python 2
def python2_gpu_ut(docker_type) {
  timeout(time: max_time, unit: 'MINUTES') {
    sh "${docker_run} ${docker_type} find . -name '*.pyc' -type f -delete"
    sh "${docker_run} ${docker_type} PYTHONPATH=./python/ nosetests --with-timer --verbose tests/python/gpu"
  }
}

// Python 3
def python3_gpu_ut(docker_type) {
  timeout(time: max_time, unit: 'MINUTES') {
    sh "${docker_run} ${docker_type} find . -name '*.pyc' -type f -delete"
    sh "${docker_run} ${docker_type} PYTHONPATH=./python/ nosetests-3.4 --with-timer --verbose tests/python/gpu"
  }
}

try {
    stage("Sanity Check") {
      timeout(time: max_time, unit: 'MINUTES') {
        node('mxnetlinux') {
          ws('workspace/sanity') {
            init_git()
            sh "python tools/license_header.py check"
            make('lint', 'cpplint rcpplint jnilint')
            make('lint', 'pylint')
          }
        }
      }
    }

    stage('Build') {
      parallel 'CPU: Openblas': {
        node('mxnetlinux') {
          ws('workspace/build-cpu') {
            init_git()
            def flag = """ \
    DEV=1                         \
    USE_PROFILER=1                \
    USE_CPP_PACKAGE=1             \
    USE_BLAS=openblas             \
    -j\$(nproc)
    """
            make("cpu", flag)
            make_test("cpu", flag)
            pack_lib('cpu')
            stash includes: build/tests/cpp/mxnet_test, name: 'cpu_cpp_ut'
          }
        }
      },

    }

    stage('Unit Test') {
      parallel 'C++: CPU': {
        node('mxnetlinux') {
          ws('workspace/ut-cpp-cpu') {
            init_git()
            unpack_lib('cpu')
            unstash 'cpu_cpp_ut'
            cpp_ut('cpu')
          }
        }
       },
      'Python2: CPU': {
        node('mxnetlinux') {
          ws('workspace/ut-python2-cpu') {
            init_git()
            unpack_lib('cpu')
            python2_ut('cpu')
          }
        }
      },
      'Python3: CPU': {
        node('mxnetlinux') {
          ws('workspace/ut-python3-cpu') {
            init_git()
            unpack_lib('cpu')
            python3_ut('cpu')
          }
        }
      },
   }
  // set build status to success at the end
  currentBuild.result = "SUCCESS"
} catch (caughtError) {
    node("mxnetlinux") {
        sh "echo caught error"
        err = caughtError
        currentBuild.result = "FAILURE"
    }
} finally {
    node("mxnetlinux") {
        // Only send email if master failed
        if (currentBuild.result == "FAILURE" && env.BRANCH_NAME == "master") {
            emailext body: 'Build for MXNet branch ${BRANCH_NAME} has broken. Please view the build at ${BUILD_URL}', replyTo: '${EMAIL}', subject: '[BUILD FAILED] Branch ${BRANCH_NAME} build ${BUILD_NUMBER}', to: '${EMAIL}'
        }
        // Remember to rethrow so the build is marked as failing
        if (err) {
            throw err
        }
    }
}
