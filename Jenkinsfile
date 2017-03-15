// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// mxnet libraries
mx_lib = 'lib/libmxnet.so, lib/libmxnet.a, dmlc-core/libdmlc.a, nnvm/lib/libnnvm.a'
// command to start a docker container
docker_run = 'tests/ci_build/ci_build.sh'
// timeout in minutes
max_time = 60

def pack_lib(name, libs=mx_lib) {
  sh """
echo "Packing ${libs} into ${name}"
echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
"""
  stash includes: libs, name: name
}

def unpack_lib(name, libs=mx_lib) {
  unstash name
  sh """
echo "Unpacked ${libs} from ${name}"
echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
"""
}

def init_git() {
  checkout scm
  retry(5) {
    timeout(time: 2, unit: 'MINUTES') {
      sh 'git submodule update --init'
    }
  }
}

def make(docker_type, make_flag) {
  timeout(time: max_time, unit: 'MINUTES') {
    try {
      echo 'Try incremental build from a previous workspace'
      sh "${docker_run} ${docker_type} make ${make_flag}"
    } catch (exc) {
      echo 'Fall back to build from scratch'
      sh "${docker_run} ${docker_type} make clean"
      sh "${docker_run} ${docker_type} make ${make_flag}"
    }
  }
}

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


// stage('Build') {
//   parallel 'CPU: openblas': {
//     timeout(time: max_time, unit: 'MINUTES') {
//       node {
//         ws('workspace/build-cpu') {
//           init_git()
//           make("${mx_run} cpu", 'USE_BLAS=openblas')
//           pack_lib('cpu', mx_lib)
//         }
//       }
//     }
//   },
//   'GPU: CUDA7.5+cuDNN5': {
//     timeout(time: max_time, unit: 'MINUTES') {
//       node('GPU') {
//         ws('workspace/build-gpu') {
//           init_git()
//           def flag = 'USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1'
//           make("${mx_run} gpu", flag)
//           pack_lib('gpu', mx_lib)
//         }
//       }
//     }
//   },
//   'Amalgamation': {
//     timeout(time: max_time, unit: 'MINUTES') {
//       node() {
//         ws('workspace/amalgamation') {
//           init_git()
//           def flag = '-C amalgamation/ USE_BLAS=openblas MIN=1'
//           make("${mx_run} cpu", flag)
//         }
//       }
//     }
//   },
//   'CPU: MKL': {
//     timeout(time: max_time, unit: 'MINUTES') {
//       node() {
//         ws('workspace/build-mkl') {
//           init_git()
//           def flag = """ \
// USE_BLAS=openblas \
// USE_MKL2017=1 \
// USE_MKL2017_EXPERIMENTAL=1 \
// MKLML_ROOT=\$(pwd)/mklml \
// -j\$(nproc)
// """
//           make("${mx_run} mkl", flag)
//           pack_lib('mkl', mx_lib)
//         }
//       }
//     }
//   }
// }

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
