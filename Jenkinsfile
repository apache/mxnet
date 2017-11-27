// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// mxnet libraries
mx_lib = 'lib/libmxnet.so, lib/libmxnet.a, dmlc-core/libdmlc.a, nnvm/lib/libnnvm.a'
// command to start a docker container
docker_run = 'tests/ci_build/ci_build.sh'
// timeout in minutes
max_time = 1440
// assign any caught errors here
err = null

// initialize source codes
def init_git() {
  deleteDir()
  retry(5) {
    try {
      timeout(time: 2, unit: 'MINUTES') {
        checkout scm
        sh 'git submodule update --init'
        sh 'git clean -d -f'        
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
      timeout(time: 2, unit: 'MINUTES') {
        checkout scm
        bat 'git submodule update --init'
        bat 'git clean -d -f'        
      }
    } catch (exc) {
      deleteDir()
      error "Failed to fetch source codes with ${exc}"
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
      sh "${docker_run} ${docker_type} --dockerbinary docker make ${make_flag}"
    } catch (exc) {
      echo 'Incremental compilation failed with ${exc}. Fall back to build from scratch'
      sh "${docker_run} ${docker_type} --dockerbinary docker sudo make clean"
      sh "${docker_run} ${docker_type} --dockerbinary docker sudo make -C amalgamation/ clean"
      sh "${docker_run} ${docker_type} --dockerbinary docker make ${make_flag}"
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

// Python unittest for CPU
// Python 2
def python2_ut(docker_type) {
  timeout(time: max_time, unit: 'MINUTES') {
    sh "${docker_run} ${docker_type} --dockerbinary docker find . -name '*.pyc' -type f -delete"
    sh "${docker_run} ${docker_type} --dockerbinary docker PYTHONPATH=./python/ nosetests-2.7 --with-timer --verbose tests/python/unittest"
    sh "${docker_run} ${docker_type} --dockerbinary docker PYTHONPATH=./python/ nosetests-2.7 --with-timer --verbose tests/python/train"
  }
}

// Python 3
def python3_ut(docker_type) {
  timeout(time: max_time, unit: 'MINUTES') {
    sh "${docker_run} ${docker_type} --dockerbinary docker find . -name '*.pyc' -type f -delete"
    sh "${docker_run} ${docker_type} --dockerbinary docker PYTHONPATH=./python/ nosetests-3.4 --with-timer --verbose tests/python/unittest"
  }
}

// GPU test has two parts. 1) run unittest on GPU, 2) compare the results on
// both CPU and GPU
// Python 2
def python2_gpu_ut(docker_type) {
  timeout(time: max_time, unit: 'MINUTES') {
    sh "${docker_run} ${docker_type} find . -name '*.pyc' -type f -delete"
    sh "${docker_run} ${docker_type} PYTHONPATH=./python/ nosetests-2.7 --with-timer --verbose tests/python/gpu"
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
        node('mxnetlinux-cpu') {
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
        node('mxnetlinux-cpu') {
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
            pack_lib('cpu')
          }
        }
      },
      'GPU: CUDA7.5+cuDNN5': {
        node('mxnetlinux-cpu') {
          ws('workspace/build-gpu') {
            init_git()
            def flag = """ \
    DEV=1                         \
    USE_PROFILER=1                \
    USE_BLAS=openblas             \
    USE_CUDA=1                    \
    USE_CUDA_PATH=/usr/local/cuda \
    USE_CUDNN=1                   \
    USE_CPP_PACKAGE=1             \
    -j\$(nproc)
    """
            make('cpu_cuda', flag)
            pack_lib('gpu')
            stash includes: 'build/cpp-package/example/test_score', name: 'cpp_test_score'
          }
        }
      },
      'Amalgamation MIN': {
        node('mxnetlinux-cpu') {
          ws('workspace/amalgamationmin') {
            init_git()
            make('cpu', '-C amalgamation/ clean')
            make('cpu', '-C amalgamation/ USE_BLAS=openblas MIN=1')
          }
        }
      },
      'Amalgamation': {
        node('mxnetlinux-cpu') {
          ws('workspace/amalgamation') {
            init_git()
            make('cpu', '-C amalgamation/ clean')
            make('cpu', '-C amalgamation/ USE_BLAS=openblas')
          }
        }
      },
      'GPU: MKLML': {
        node('mxnetlinux-cpu') {
          ws('workspace/build-mklml') {
            init_git()
            def flag = """ \
    DEV=1                         \
    USE_PROFILER=1                \
    USE_BLAS=openblas             \
    USE_MKL2017=1                 \
    USE_MKL2017_EXPERIMENTAL=1    \
    USE_CUDA=1                    \
    USE_CUDA_PATH=/usr/local/cuda \
    USE_CUDNN=1                   \
    USE_CPP_PACKAGE=1             \
    -j\$(nproc)
    """
            make('cpu_cuda', flag) #TODO: Check, this should use MKLML
            pack_lib('mklml')
          }
        }
      }
    }

    stage('Unit Test') {
      parallel 'Python2: CPU': {
        node('mxnetlinux-cpu') {
          ws('workspace/ut-python2-cpu') {
            init_git()
            unpack_lib('cpu')
            python2_ut('cpu')
          }
        }
      },
      'Python3: CPU': {
        node('mxnetlinux-cpu') {
          ws('workspace/ut-python3-cpu') {
            init_git()
            unpack_lib('cpu')
            python3_ut('cpu')
          }
        }
      },
      'Python2: GPU': {
        node('mxnetlinux-gpu') {
          ws('workspace/ut-python2-gpu') {
            init_git()
            unpack_lib('gpu', mx_lib)
            python2_gpu_ut('gpu')
          }
        }
      },
      'Python3: GPU': {
        node('mxnetlinux-gpu') {
          ws('workspace/ut-python3-gpu') {
            init_git()
            unpack_lib('gpu', mx_lib)
            python3_gpu_ut('gpu')
          }
        }
      },
      'Python2: MKLML-CPU': {
        node('mxnetlinux-cpu') {
          ws('workspace/ut-python2-mklml-cpu') {
            init_git()
            unpack_lib('mklml')
            python2_ut('cpu_cuda') #CHECK: Rethink, this should use MKLML
          }
        }
      },
      'Python2: MKLML-GPU': {
        node('mxnetlinux-gpu') {
          ws('workspace/ut-python2-mklml-gpu') {
            init_git()
            unpack_lib('mklml')
            python2_gpu_ut('mklml_gpu')
          }
        }
      },
      'Python3: MKLML-CPU': {
        node('mxnetlinux-cpu') {
          ws('workspace/ut-python3-mklml-cpu') {
            init_git()
            unpack_lib('mklml')
            python3_ut('cpu_cuda') #TODO: Check, this should use MKLML
          }
        }
      },
      'Python3: MKLML-GPU': {
        node('mxnetlinux-gpu') {
          ws('workspace/ut-python3-mklml-gpu') {
            init_git()
            unpack_lib('mklml')
            python3_gpu_ut('mklml_gpu')
          }
        }
      },
      'Scala: CPU': {
        node('mxnetlinux-cpu') {
          ws('workspace/ut-scala-cpu') {
            init_git()
            unpack_lib('cpu')
            timeout(time: max_time, unit: 'MINUTES') {
              sh "${docker_run} cpu make scalapkg USE_BLAS=openblas"
              sh "${docker_run} cpu make scalatest USE_BLAS=openblas"
            }
          }
        }
      },
      'Perl: CPU': {
            node('mxnetlinux-cpu') {
                ws('workspace/ut-perl-cpu') {
                    init_git()
                    unpack_lib('cpu')
                    timeout(time: max_time, unit: 'MINUTES') {
                        sh "${docker_run} cpu ./perl-package/test.sh"
                    }
                }
            }
      },
      'Perl: GPU': {
            node('mxnetlinux-gpu') {
                ws('workspace/ut-perl-gpu') {
                    init_git()
                    unpack_lib('gpu')
                    timeout(time: max_time, unit: 'MINUTES') {
                        sh "${docker_run} gpu ./perl-package/test.sh"
                    }
                }
            }
      },
      'R: CPU': {
        node('mxnetlinux-cpu') {
          ws('workspace/ut-r-cpu') {
            init_git()
            unpack_lib('cpu')
            timeout(time: max_time, unit: 'MINUTES') {
              sh "${docker_run} cpu rm -rf .Renviron"
              sh "${docker_run} cpu mkdir -p /workspace/ut-r-cpu/site-library"
              sh "${docker_run} cpu make rpkg USE_BLAS=openblas R_LIBS=/workspace/ut-r-cpu/site-library"
              sh "${docker_run} cpu R CMD INSTALL --library=/workspace/ut-r-cpu/site-library R-package"
              sh "${docker_run} cpu make rpkgtest R_LIBS=/workspace/ut-r-cpu/site-library"
            }
          }
        }
      },
      'R: GPU': {
        node('mxnetlinux-gpu') {
          ws('workspace/ut-r-gpu') {
            init_git()
            unpack_lib('gpu')
            timeout(time: max_time, unit: 'MINUTES') {
              sh "${docker_run} gpu rm -rf .Renviron"
              sh "${docker_run} gpu mkdir -p /workspace/ut-r-gpu/site-library"
              sh "${docker_run} gpu make rpkg USE_BLAS=openblas R_LIBS=/workspace/ut-r-gpu/site-library"
              sh "${docker_run} gpu R CMD INSTALL --library=/workspace/ut-r-gpu/site-library R-package"
              sh "${docker_run} gpu make rpkgtest R_LIBS=/workspace/ut-r-gpu/site-library R_GPU_ENABLE=1"
            }
          }
        }
      }
    }

    stage('Integration Test') {
      parallel 'Python GPU': {
        node('mxnetlinux-gpu') {
          ws('workspace/it-python-gpu') {
            init_git()
            unpack_lib('gpu')
            timeout(time: max_time, unit: 'MINUTES') {
              sh "${docker_run} gpu --dockerbinary nvidia-docker PYTHONPATH=./python/ python example/image-classification/test_score.py"
            }
          }
        }
      },
      'Caffe GPU': {
        node('mxnetlinux-gpu') {
          ws('workspace/it-caffe') {
            init_git()
            unpack_lib('gpu')
            timeout(time: max_time, unit: 'MINUTES') {
              sh "${docker_run} caffe_gpu --dockerbinary nvidia-docker PYTHONPATH=/caffe/python:./python python tools/caffe_converter/test_converter.py"
            }
          }
        }
      },
      'cpp-package GPU': {
        node('mxnetlinux-gpu') {
          ws('workspace/it-cpp-package') {
            init_git()
            unpack_lib('gpu')
            unstash 'cpp_test_score'
            timeout(time: max_time, unit: 'MINUTES') {
              sh "${docker_run} gpu --dockerbinary nvidia-docker cpp-package/tests/ci_test.sh"
            }
          }
        }
      }
    }

    stage('Deploy') {
      node('mxnetlinux-cpu') {
        ws('workspace/docs') {
          if (env.BRANCH_NAME == "master") {
            init_git()
            sh "make clean"
            sh "make docs"
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
