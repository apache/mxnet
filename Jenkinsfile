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
mx_lib = 'lib/libmxnet.so, lib/libmxnet.a, dmlc-core/libdmlc.a, nnvm/lib/libnnvm.a'
// mxnet cmake libraries, in cmake builds we do not produce a libnvvm static library by default.
mx_cmake_lib = 'build/libmxnet.so, build/libmxnet.a, build/dmlc-core/libdmlc.a, build/tests/mxnet_unit_tests, build/3rdparty/openmp/runtime/src/libomp.so'
mx_cmake_mkldnn_lib = 'build/libmxnet.so, build/libmxnet.a, build/dmlc-core/libdmlc.a, build/tests/mxnet_unit_tests, build/3rdparty/openmp/runtime/src/libomp.so, build/3rdparty/mkldnn/src/libmkldnn.so, build/3rdparty/mkldnn/src/libmkldnn.so.0'
mx_mkldnn_lib = 'lib/libmxnet.so, lib/libmxnet.a, lib/libiomp5.so, lib/libmklml_gnu.so, lib/libmkldnn.so, lib/libmkldnn.so.0, lib/libmklml_intel.so, dmlc-core/libdmlc.a, nnvm/lib/libnnvm.a'
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

def init_git_win() {
  deleteDir()
  retry(5) {
    try {
      // Make sure wait long enough for api.github.com request quota. Important: Don't increase the amount of
      // retries as this will increase the amount of requests and worsen the throttling
      timeout(time: 15, unit: 'MINUTES') {
        checkout scm
        bat 'git submodule update --init --recursive'
        bat 'git clean -d -f'
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

// Python unittest for CPU
// Python 2
def python2_ut(docker_container_name) {
  timeout(time: max_time, unit: 'MINUTES') {
    sh "ci/build.py --build --platform ${docker_container_name} /work/runtime_functions.sh unittest_ubuntu_python2_cpu"
  }
}

// Python 3
def python3_ut(docker_container_name) {
  timeout(time: max_time, unit: 'MINUTES') {
    sh "ci/build.py --build --platform ${docker_container_name} /work/runtime_functions.sh unittest_ubuntu_python3_cpu"
  }
}

// GPU test has two parts. 1) run unittest on GPU, 2) compare the results on
// both CPU and GPU
// Python 2
def python2_gpu_ut(docker_container_name) {
  timeout(time: max_time, unit: 'MINUTES') {
    sh "ci/build.py --nvidiadocker --build --platform ${docker_container_name} /work/runtime_functions.sh unittest_ubuntu_python2_gpu"
  }
}

// Python 3
def python3_gpu_ut(docker_container_name) {
  timeout(time: max_time, unit: 'MINUTES') {
    sh "ci/build.py --nvidiadocker --build --platform ${docker_container_name} /work/runtime_functions.sh unittest_ubuntu_python3_gpu"
  }
}

try {
  stage("Sanity Check") {
    node('mxnetlinux-cpu') {
      ws('workspace/sanity') {
        init_git()
        sh "ci/build.py --build --platform ubuntu_cpu /work/runtime_functions.sh sanity_check"
      }
    }
  }

  stage('Build') {
    parallel 'CPU: CentOS 7': {
      node('mxnetlinux-cpu') {
        ws('workspace/build-centos7-cpu') {
          init_git()
          sh "ci/build.py --build --platform centos7_cpu /work/runtime_functions.sh build_centos7_cpu"
          pack_lib('centos7_cpu')
        }
      }
    },
    'GPU: CentOS 7': {
      node('mxnetlinux-cpu') {
        ws('workspace/build-centos7-gpu') {
          init_git()
          sh "ci/build.py --build --platform centos7_gpu /work/runtime_functions.sh build_centos7_gpu"
          pack_lib('centos7_gpu')
        }
      }
    },
    'CPU: Openblas': {
      node('mxnetlinux-cpu') {
        ws('workspace/build-cpu-openblas') {
          init_git()
          sh "ci/build.py --build --platform ubuntu_cpu /work/runtime_functions.sh build_ubuntu_cpu_openblas"
          pack_lib('cpu')
        }
      }
    },
    'CPU: Clang 3.9': {
      node('mxnetlinux-cpu') {
        ws('workspace/build-cpu-clang39') {
          init_git()
          sh "ci/build.py --build --platform ubuntu_cpu /work/runtime_functions.sh build_ubuntu_cpu_clang39"
        }
      }
    },
    'CPU: Clang 5': {
      node('mxnetlinux-cpu') {
        ws('workspace/build-cpu-clang50') {
          init_git()
          sh "ci/build.py --build --platform ubuntu_cpu /work/runtime_functions.sh build_ubuntu_cpu_clang50"
        }
      }
    },
    'CPU: MKLDNN': {
      node('mxnetlinux-cpu') {
        ws('workspace/build-mkldnn-cpu') {
          init_git()
          sh "ci/build.py --build --platform ubuntu_cpu /work/runtime_functions.sh build_ubuntu_cpu_mkldnn"
          pack_lib('mkldnn_cpu', mx_mkldnn_lib)
        }
      }
    },
    'GPU: MKLDNN': {
      node('mxnetlinux-cpu') {
        ws('workspace/build-mkldnn-gpu') {
          init_git()
          sh "ci/build.py --build --platform ubuntu_build_cuda /work/runtime_functions.sh build_ubuntu_gpu_mkldnn"
          pack_lib('mkldnn_gpu', mx_mkldnn_lib)
        }
      }
    },
    'GPU: CUDA8.0+cuDNN5': {
      node('mxnetlinux-cpu') {
        ws('workspace/build-gpu') {
          init_git()
          sh "ci/build.py --build --platform ubuntu_build_cuda /work/runtime_functions.sh build_ubuntu_gpu_cuda8_cudnn5" 
          pack_lib('gpu')
          stash includes: 'build/cpp-package/example/test_score', name: 'cpp_test_score'
          stash includes: 'build/cpp-package/example/test_optimizer', name: 'cpp_test_optimizer'
        }
      }
    },
    'Amalgamation MIN': {
      node('mxnetlinux-cpu') {
        ws('workspace/amalgamationmin') {
          init_git()
          sh "ci/build.py --build --platform ubuntu_cpu /work/runtime_functions.sh build_ubuntu_amalgamation_min"
        }
      }
    },
    'Amalgamation': {
      node('mxnetlinux-cpu') {
        ws('workspace/amalgamation') {
          init_git()
          sh "ci/build.py --build --platform ubuntu_cpu /work/runtime_functions.sh build_ubuntu_amalgamation"
        }
      }
    },

    'GPU: CMake MKLDNN': {
      node('mxnetlinux-cpu') {
        ws('workspace/build-cmake-mkldnn-gpu') {
          init_git()
          sh "ci/build.py --build --platform ubuntu_gpu /work/runtime_functions.sh build_ubuntu_gpu_cmake_mkldnn" //build_cuda
          pack_lib('cmake_mkldnn_gpu', mx_cmake_mkldnn_lib)
        }
      }
    },
    'GPU: CMake': {
      node('mxnetlinux-cpu') {
        ws('workspace/build-cmake-gpu') {
          init_git()
          sh "ci/build.py --build --platform ubuntu_gpu /work/runtime_functions.sh build_ubuntu_gpu_cmake" //build_cuda
          pack_lib('cmake_gpu', mx_cmake_lib)
        }
      }
    },
    'Build CPU windows':{
      node('mxnetwindows-cpu') {
        timeout(time: max_time, unit: 'MINUTES') {
          ws('workspace/build-cpu') {
            withEnv(['OpenBLAS_HOME=C:\\mxnet\\openblas', 'OpenCV_DIR=C:\\mxnet\\opencv_vc14', 'CUDA_PATH=C:\\CUDA\\v8.0']) {
              init_git_win()
              bat """mkdir build_vc14_cpu
                call "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\bin\\x86_amd64\\vcvarsx86_amd64.bat"
                cd build_vc14_cpu
                cmake -G \"Visual Studio 14 2015 Win64\" -DUSE_CUDA=0 -DUSE_CUDNN=0 -DUSE_NVRTC=0 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_PROFILER=1 -DUSE_BLAS=open -DUSE_LAPACK=1 -DUSE_DIST_KVSTORE=0 ${env.WORKSPACE}"""
              bat 'C:\\mxnet\\build_vc14_cpu.bat'

              bat '''rmdir /s/q pkg_vc14_cpu
                mkdir pkg_vc14_cpu\\lib
                mkdir pkg_vc14_cpu\\python
                mkdir pkg_vc14_cpu\\include
                mkdir pkg_vc14_cpu\\build
                copy build_vc14_cpu\\Release\\libmxnet.lib pkg_vc14_cpu\\lib
                copy build_vc14_cpu\\Release\\libmxnet.dll pkg_vc14_cpu\\build
                xcopy python pkg_vc14_cpu\\python /E /I /Y
                xcopy include pkg_vc14_cpu\\include /E /I /Y
                xcopy dmlc-core\\include pkg_vc14_cpu\\include /E /I /Y
                xcopy mshadow\\mshadow pkg_vc14_cpu\\include\\mshadow /E /I /Y
                xcopy nnvm\\include pkg_vc14_cpu\\nnvm\\include /E /I /Y
                del /Q *.7z
                7z.exe a vc14_cpu.7z pkg_vc14_cpu\\
                '''
              stash includes: 'vc14_cpu.7z', name: 'vc14_cpu'
            }
          }
        }
      }
    },
    //Todo: Set specific CUDA_ARCh for windows builds in cmake
    'Build GPU windows':{
      node('mxnetwindows-cpu') {
        timeout(time: max_time, unit: 'MINUTES') {
          ws('workspace/build-gpu') {
            withEnv(['OpenBLAS_HOME=C:\\mxnet\\openblas', 'OpenCV_DIR=C:\\mxnet\\opencv_vc14', 'CUDA_PATH=C:\\CUDA\\v8.0']) {
            init_git_win()
            bat """mkdir build_vc14_gpu
              call "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\bin\\x86_amd64\\vcvarsx86_amd64.bat"
              cd build_vc14_gpu
              cmake -G \"NMake Makefiles JOM\" -DUSE_CUDA=1 -DUSE_CUDNN=1 -DUSE_NVRTC=1 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_PROFILER=1 -DUSE_BLAS=open -DUSE_LAPACK=1 -DUSE_DIST_KVSTORE=0 -DCUDA_ARCH_NAME=All -DCMAKE_CXX_FLAGS_RELEASE="/FS /MD /O2 /Ob2 /DNDEBUG" -DCMAKE_BUILD_TYPE=Release ${env.WORKSPACE}"""
            bat 'C:\\mxnet\\build_vc14_gpu.bat'
            bat '''rmdir /s/q pkg_vc14_gpu
              mkdir pkg_vc14_gpu\\lib
              mkdir pkg_vc14_gpu\\python
              mkdir pkg_vc14_gpu\\include
              mkdir pkg_vc14_gpu\\build
              copy build_vc14_gpu\\libmxnet.lib pkg_vc14_gpu\\lib
              copy build_vc14_gpu\\libmxnet.dll pkg_vc14_gpu\\build
              xcopy python pkg_vc14_gpu\\python /E /I /Y
              xcopy include pkg_vc14_gpu\\include /E /I /Y
              xcopy dmlc-core\\include pkg_vc14_gpu\\include /E /I /Y
              xcopy mshadow\\mshadow pkg_vc14_gpu\\include\\mshadow /E /I /Y
              xcopy nnvm\\include pkg_vc14_gpu\\nnvm\\include /E /I /Y
              del /Q *.7z
              7z.exe a vc14_gpu.7z pkg_vc14_gpu\\
              '''
            stash includes: 'vc14_gpu.7z', name: 'vc14_gpu'
            }
          }
        }
      }
    },
    'NVidia Jetson / ARMv8':{
      node('mxnetlinux-cpu') {
        ws('workspace/build-jetson-armv8') {
          init_git()
          sh "ci/build.py --build --platform jetson /work/runtime_functions.sh build_jetson"
        }
      }
    },
    'Raspberry / ARMv7':{
      node('mxnetlinux-cpu') {
        ws('workspace/build-raspberry-armv7') {
          init_git()
          sh "ci/build.py --build --platform armv7 /work/runtime_functions.sh build_armv7"
        }
      }
    }
  } // End of stage('Build')

  stage('Unit Test') {
    parallel 'Python2: CPU': {
      node('mxnetlinux-cpu') {
        ws('workspace/ut-python2-cpu') {
          init_git()
          unpack_lib('cpu')
          python2_ut('ubuntu_cpu')
        }
      }
    },
    'Python3: CPU': {
      node('mxnetlinux-cpu') {
        ws('workspace/ut-python3-cpu') {
          init_git()
          unpack_lib('cpu')
          python3_ut('ubuntu_cpu')
        }
      }
    },
    'Python2: GPU': {
      node('mxnetlinux-gpu') {
        ws('workspace/ut-python2-gpu') {
          init_git()
          unpack_lib('gpu', mx_lib)
          python2_gpu_ut('ubuntu_gpu')
        }
      }
    },
    'Python3: GPU': {
      node('mxnetlinux-gpu') {
        ws('workspace/ut-python3-gpu') {
          init_git()
          unpack_lib('gpu', mx_lib)
          python3_gpu_ut('ubuntu_gpu')
        }
      }
    },
    'Python2: MKLDNN-CPU': {
      node('mxnetlinux-cpu') {
        ws('workspace/ut-python2-mkldnn-cpu') {
          init_git()
          unpack_lib('mkldnn_cpu', mx_mkldnn_lib)
          python2_ut('ubuntu_cpu')
        }
      }
    },
    'Python2: MKLDNN-GPU': {
      node('mxnetlinux-gpu') {
        ws('workspace/ut-python2-mkldnn-gpu') {
          init_git()
          unpack_lib('mkldnn_gpu', mx_mkldnn_lib)
          python2_gpu_ut('ubuntu_gpu')
        }
      }
    },
    'Python3: MKLDNN-CPU': {
      node('mxnetlinux-cpu') {
        ws('workspace/ut-python3-mkldnn-cpu') {
          init_git()
          unpack_lib('mkldnn_cpu', mx_mkldnn_lib)
          python3_ut('ubuntu_cpu')
        }
      }
    },
    'Python3: MKLDNN-GPU': {
      node('mxnetlinux-gpu') {
        ws('workspace/ut-python3-mkldnn-gpu') {
          init_git()
          unpack_lib('mkldnn_gpu', mx_mkldnn_lib)
          python3_gpu_ut('ubuntu_gpu')
        }
      }
    },
    'Python3: CentOS 7 CPU': {
      node('mxnetlinux-cpu') {
        ws('workspace/build-centos7-cpu') {
          init_git()
          unpack_lib('centos7_cpu')
          timeout(time: max_time, unit: 'MINUTES') {
            sh "ci/build.py --build --platform centos7_cpu /work/runtime_functions.sh unittest_centos7_cpu"
          }
        }
      }
    },
    'Python3: CentOS 7 GPU': {
      node('mxnetlinux-gpu') {
        ws('workspace/build-centos7-gpu') {
          init_git()
          unpack_lib('centos7_gpu')
          timeout(time: max_time, unit: 'MINUTES') {
            sh "ci/build.py --nvidiadocker --build --platform centos7_gpu /work/runtime_functions.sh unittest_centos7_gpu"
          }
        }
      }
    },
    'Scala: CPU': {
      node('mxnetlinux-cpu') {
        ws('workspace/ut-scala-cpu') {
          init_git()
          unpack_lib('cpu')
          timeout(time: max_time, unit: 'MINUTES') {
            sh "ci/build.py --build --platform ubuntu_cpu /work/runtime_functions.sh unittest_ubuntu_cpu_scala"
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
            sh "ci/build.py --build --platform ubuntu_cpu /work/runtime_functions.sh unittest_ubuntu_cpugpu_perl"
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
            sh "ci/build.py --nvidiadocker --build --platform ubuntu_gpu /work/runtime_functions.sh unittest_ubuntu_cpugpu_perl"
          }
        }
      }
    },
    'Cpp: GPU': {
      node('mxnetlinux-gpu') {
        ws('workspace/ut-cpp-gpu') {
          init_git()
          unpack_lib('cmake_gpu', mx_cmake_lib)
          timeout(time: max_time, unit: 'MINUTES') {
            sh "ci/build.py --nvidiadocker --build --platform ubuntu_gpu /work/runtime_functions.sh unittest_ubuntu_gpu_cpp"
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
            sh "ci/build.py --build --platform ubuntu_cpu /work/runtime_functions.sh unittest_ubuntu_cpu_R"
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
            sh "ci/build.py --nvidiadocker --build --platform ubuntu_gpu /work/runtime_functions.sh unittest_ubuntu_gpu_R"
          }
        }
      }
    },

    'Python 2: CPU Win':{
      node('mxnetwindows-cpu') {
        timeout(time: max_time, unit: 'MINUTES') {
          ws('workspace/ut-python-cpu') {
            init_git_win()
            unstash 'vc14_cpu'
            bat '''rmdir /s/q pkg_vc14_cpu
              7z x -y vc14_cpu.7z'''
            bat """xcopy C:\\mxnet\\data data /E /I /Y
              xcopy C:\\mxnet\\model model /E /I /Y
              call activate py2
              set PYTHONPATH=${env.WORKSPACE}\\pkg_vc14_cpu\\python
              del /S /Q ${env.WORKSPACE}\\pkg_vc14_cpu\\python\\*.pyc
              C:\\mxnet\\test_cpu.bat"""
          }
        }
      }
    },
    'Python 3: CPU Win': {
      node('mxnetwindows-cpu') {
        timeout(time: max_time, unit: 'MINUTES') {
          ws('workspace/ut-python-cpu') {
            init_git_win()
            unstash 'vc14_cpu'
            bat '''rmdir /s/q pkg_vc14_cpu
              7z x -y vc14_cpu.7z'''
            bat """xcopy C:\\mxnet\\data data /E /I /Y
              xcopy C:\\mxnet\\model model /E /I /Y
              call activate py3
              set PYTHONPATH=${env.WORKSPACE}\\pkg_vc14_cpu\\python
              del /S /Q ${env.WORKSPACE}\\pkg_vc14_cpu\\python\\*.pyc
              C:\\mxnet\\test_cpu.bat"""
          }
        }
      }
    },
    'Python 2: GPU Win':{
      node('mxnetwindows-gpu') {
        timeout(time: max_time, unit: 'MINUTES') {
          ws('workspace/ut-python-gpu') {
            init_git_win()
            unstash 'vc14_gpu'
            bat '''rmdir /s/q pkg_vc14_gpu
              7z x -y vc14_gpu.7z'''
            bat """xcopy C:\\mxnet\\data data /E /I /Y
              xcopy C:\\mxnet\\model model /E /I /Y
              call activate py2
              set PYTHONPATH=${env.WORKSPACE}\\pkg_vc14_gpu\\python
              del /S /Q ${env.WORKSPACE}\\pkg_vc14_gpu\\python\\*.pyc
              C:\\mxnet\\test_gpu.bat"""
          }
        }
      }
    },
    'Python 3: GPU Win':{
      node('mxnetwindows-gpu') {
        timeout(time: max_time, unit: 'MINUTES') {
          ws('workspace/ut-python-gpu') {
          init_git_win()
          unstash 'vc14_gpu'
          bat '''rmdir /s/q pkg_vc14_gpu
            7z x -y vc14_gpu.7z'''
          bat """xcopy C:\\mxnet\\data data /E /I /Y
            xcopy C:\\mxnet\\model model /E /I /Y
            call activate py3
            set PYTHONPATH=${env.WORKSPACE}\\pkg_vc14_gpu\\python
            del /S /Q ${env.WORKSPACE}\\pkg_vc14_gpu\\python\\*.pyc
            C:\\mxnet\\test_gpu.bat"""
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
            sh "ci/build.py --nvidiadocker --build --platform ubuntu_gpu /work/runtime_functions.sh integrationtest_ubuntu_gpu_python"
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
            sh "ci/build.py --nvidiadocker --build --platform ubuntu_gpu /work/runtime_functions.sh integrationtest_ubuntu_gpu_caffe"
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
          unstash 'cpp_test_optimizer'
          timeout(time: max_time, unit: 'MINUTES') {
            sh "ci/build.py --nvidiadocker --build --platform ubuntu_gpu /work/runtime_functions.sh integrationtest_ubuntu_gpu_cpp_package"
          }
        }
      }
    }
  }

  stage('Deploy') {
    node('mxnetlinux-cpu') {
      ws('workspace/docs') {
        init_git()
        timeout(time: max_time, unit: 'MINUTES') {
          sh "ci/build.py --build --platform ubuntu_cpu /work/runtime_functions.sh deploy_docs"
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
