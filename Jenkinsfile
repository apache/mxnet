// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

def mx_lib = 'lib/libmxnet.so, lib/libmxnet.a, dmlc-core/libdmlc.a, nnvm/lib/libnnvm.a'
def mx_run = 'tests/ci_build/ci_build.sh'

def pack_lib(name, mx_lib) {
  sh """
echo "Packing ${mx_lib} into ${name}"
echo ${mx_lib} | sed -e 's/,/ /g' | xargs md5sum
"""
  stash includes: mx_lib, name: name
}

def unpack_lib(name, mx_lib) {
  unstash name
  sh """
echo "Unpacked ${mx_lib} from ${name}"
echo ${mx_lib} | sed -e 's/,/ /g' | xargs md5sum
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

def make(docker_run, make_flag) {
  try {
    echo 'Try incremental build from a previous workspace'
    sh "${docker_run} make ${make_flag}"
  } catch (exc) {
    echo 'Fall back to build from scratch'
    sh "${docker_run} make clean"
    sh "${docker_run} make ${make_flag}"
  }
}

stage('Build') {
  node {
    ws('workspace/build-mkl') {
      init_git()
      def flag = " USE_MKL2017=1 USE_MKL2017_EXPERIMENTAL=1 MKLML_ROOT=\$(pwd) ADD_CFLAGS=-I\$(pwd)/include -j\$(nproc)"
      make("${mx_run} cpu", flag)
    }
  }
}
