// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// mxnet libraries
mx_lib = 'lib/libmxnet.so, lib/libmxnet.a, dmlc-core/libdmlc.a, nnvm/lib/libnnvm.a'
// command to start a docker container
docker_run = 'tests/ci_build/ci_build.sh'
// timeout in minutes
max_time = 60

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

// initialize source codes
def init_git() {
  checkout scm
  retry(5) {
    timeout(time: 2, unit: 'MINUTES') {
      sh 'git submodule update --init'
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
      sh "${docker_run} ${docker_type} make clean"
      sh "${docker_run} ${docker_type} make ${make_flag}"
    }
  }
}

// Python unittest
def python_ut(docker_type) {
  timeout(time: max_time, unit: 'MINUTES') {
    sh "${docker_run} ${docker_type} PYTHONPATH=./python/ nosetests --with-timer --verbose tests/python/unittest"
    sh "${docker_run} ${docker_type} PYTHONPATH=./python/ nosetests-3.4 --with-timer --verbose tests/python/unittest"
  }
}

stage("Sanity Check") {
  timeout(time: max_time, unit: 'MINUTES') {
    node {
      ws('workspace/sanity') {
        init_git()
        make('lint', 'cpplint rcpplint jnilint')
        make('lint', 'pylint')
      }
    }
  }
}


stage('Build') {
  parallel 'CPU: openblas': {
    node {
      ws('workspace/build-cpu') {
        init_git()
        make("cpu", "USE_BLAS=openblas -j\$(nproc)")
        pack_lib('cpu')
      }
    }
  },
  'GPU: CUDA7.5+cuDNN5': {
    node('GPU') {
      ws('workspace/build-gpu') {
        init_git()
        def flag = """ \
USE_BLAS=openblas             \
USE_CUDA=1                    \
USE_CUDA_PATH=/usr/local/cuda \
USE_CUDNN=1                   \
-j\$(nproc)
"""
        make('gpu', flag)
        pack_lib('gpu')
      }
    }
  },
  'Amalgamation': {
    node() {
      ws('workspace/amalgamation') {
        init_git()
        make('cpu', '-C amalgamation/ USE_BLAS=openblas MIN=1')
      }
    }
  },
  'CPU: MKL': {
    node() {
      ws('workspace/build-mkl') {
        init_git()
        def flag = """ \
USE_BLAS=openblas          \
USE_MKL2017=1              \
USE_MKL2017_EXPERIMENTAL=1 \
MKLML_ROOT=\$(pwd)/mklml   \
-j\$(nproc)
"""
        make('mkl', flag)
        pack_lib('mkl')
      }
    }
  }
}

// stage('Unit Test') {
//   parallel 'Python2/3: CPU': {
//     timeout(time: max_time, unit: 'MINUTES') {
//       node {
//         ws('workspace/ut-python-cpu') {
//           init_git()
//           unpack_lib('cpu', mx_lib)
//           sh "${mx_run} cpu 'PYTHONPATH=./python/ nosetests --with-timer --verbose tests/python/unittest'"
//           sh "${mx_run} cpu 'PYTHONPATH=./python/ nosetests-3.4 --with-timer --verbose tests/python/unittest'"
//         }
//       }
//     }
//   },
//   'Python2/3: GPU': {
//       node('GPU') {
//         ws('workspace/ut-python-gpu') {
//           init_git()
//           unpack_lib('gpu', mx_lib)
//         }
//       }
//     }
//   },
//   'Python2/3: MKL': {
//     timeout(time: max_time, unit: 'MINUTES') {
//       node {
//         ws('workspace/ut-python-mkl') {
//         init_git()
//         unpack_lib('mkl', mx_lib)
//         sh "${mx_run} mkl PYTHONPATH=./python/ nosetests --with-timer --verbose tests/python/unittest"
//         sh "${mx_run} mkl PYTHONPATH=./python/ nosetests-3.4 --with-timer --verbose tests/python/unittest"
//         }
//       }
//     }
//   },
//   'Scala: CPU': {
//     timeout(time: max_time, unit: 'MINUTES') {
//       node {
//         ws('workspace/ut-scala-cpu') {
//           init_git()
//           unpack_lib 'cpu', mx_lib
//           sh "${mx_run} cpu make scalapkg USE_BLAS=openblas"
//           sh "${mx_run} cpu make scalatest USE_BLAS=openblas"
//         }
//       }
//     }
//   }
// }
