stage("Sanity Check") {
  node('master') {
    checkout scm
    sh 'git submodule update --init'
    sh 'tests/ci_build/ci_build.sh lint make cpplint'
    sh 'tests/ci_build/ci_build.sh lint make rcpplint'
    sh 'tests/ci_build/ci_build.sh lint make jnilint'
    sh 'tests/ci_build/ci_build.sh lint make pylint'
  }
}
stage('Build') {
  parallel 'CPU': {
    node {
      checkout scm
      sh 'git submodule update --init'     
      sh '''echo "cpu hahaha" >lib/mx.a
      '''
      stash includes: 'lib/mx.*', name: 'cpu'
      echo "CPU Build"
    }
  },
  'CUDA 7.5+cuDNN5': {
    node('GPU') {
      checkout scm
      sh 'git submodule update --init'     
//      sh '''tests/ci_build/ci_build.sh gpu make -j$(nproc) \
//USE_CUDA=1 \
//USE_CUDA_PATH=/usr/local/cuda \
//USE_CUDNN=1 \
//USE_BLAS=openblas \
//EXTRA_OPERATORS=example/ssd/operator 
//      '''
      sh 'echo "gpu hehehe" >lib/mx.b'
      stash includes: 'lib/mx.*', name: 'gpu'
    }
  },
  'CUDA 8+cuDNN5': {
    node {
      echo "yy"
    }
  },
  'MKL' : {
    node {
      catchError {
        sh 'exit 1'
      }
    }
  },
  'Amalgamation': {
    node {
      try {
        sh 'exit 1'
      }
      catch (exc) {
        echo 'Something failed, I should sound the klaxons!'
      }
      echo "xxx"
    }
  }
}

stage('Unit Test') {
  parallel 'Python2': {
    node {
      echo "test"
      sh "ls"
      sh "pwd"
      unstash 'gpu'
      sh 'cat lib/mx.*'
    }
  },
  'Python3': {
    node {
      echo "test"
    }
  },
  'Scala': {
    node {
      echo "xxx"
      unstash 'cpu'
      sh 'cat lib/mx.*'
    }
  }
}
