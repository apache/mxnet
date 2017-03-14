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

stage("Sanity Check") {
  node {
    ws('workspace/sanity') {
      init_git()
      sh "${mx_run} lint make cpplint"
      sh "${mx_run} lint make rcpplint"
      sh "${mx_run} lint make jnilint"
      sh "${mx_run} lint make pylint"
    }
  }
}


stage('Build') {
  parallel 'CPU': {
    node {
      ws('workspace/build-cpu') {
        init_git()
        def flag = 'USE_BLAS=openblas'
        try {
          echo 'Try incremental build from a previous workspace'
          sh "${mx_run} cpu make -j\$(nproc) ${flag}"
        } catch (exc) {
          echo 'Fall back to build from scratch'
          sh "${mx_run} cpu make clean"
          sh "${mx_run} cpu make -j\$(nproc) ${flag}"
        }
        pack_lib 'cpu', mx_lib
      }
    }
  },
  'GPU: CUDA7.5+cuDNN5': {
    node('GPU') {
      ws('workspace/build-gpu') {
        init_git()
        def flag = 'USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1'
        try {
          echo 'Try incremental build from a previous workspace'
          sh "${mx_run} gpu make -j\$(nproc) ${flag}"
        } catch (exc) {
          echo 'Fall back to build from scratch'
          sh "${mx_run} gpu make clean"
          sh "${mx_run} gpu make -j\$(nproc) ${flag}"
        }
        pack_lib 'gpu', mx_lib
      }
    }
  },
  'Amalgamation': {
    node() {
      ws('workspace/amalgamation') {
        init_git()
        def flag = '-C amalgamation/ USE_BLAS=openblas MIN=1'
        sh "${mx_run} cpu make ${flag}"
      }
    }
  }
}

stage('Unit Test') {
  parallel 'Python2/3: CPU': {
    node {
      ws('workspace/ut-python-cpu') {
        init_git()
        unpack_lib 'cpu', mx_lib
        sh "${mx_run} cpu 'PYTHONPATH=./python/ nosetests --with-timer --verbose tests/python/unittest'"
        sh "${mx_run} cpu 'PYTHONPATH=./python/ nosetests-3.4 --with-timer --verbose tests/python/unittest'"
      }
    }
  },
  'Python2/3: GPU': {
    node('GPU') {
      ws('workspace/ut-python-gpu') {
        init_git()
        unpack_lib 'gpu', mx_lib
        sh "${mx_run} gpu 'PYTHONPATH=./python/ nosetests --with-timer --verbose tests/python/unittest'"
        sh "${mx_run} gpu 'PYTHONPATH=./python/ nosetests-3.4 --with-timer --verbose tests/python/unittest'"
      }
    }
  },
  'Scala: CPU': {
    node {
      ws('workspace/ut-scala-cpu') {
        init_git()
        unpack_lib 'cpu', mx_lib
        sh "${mx_run} cpu make scalapkg USE_BLAS=openblas"
        sh "${mx_run} cpu make scalatest USE_BLAS=openblas"
      }
    }
  }
}
