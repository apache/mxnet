stage("Sanity Check") {
  node('master') {
    echo "hello world"
    sh 'tests/ci_build/ci_build.sh lint pylint python/mxnet --rcfile=./tests/ci_build/pylintrc -r y'
  }
}
stage('Build') {
  parallel 'CPU': {
    node {
      echo "CPU Build"
    }
  },
  'CUDA 7.5+cuDNN5': {
    node {
      echo "xxx"
    }
  },
  'CUDA 8+cuDNN5': {
    node {
      echo "yy"
    }
  },
  'MKL' : {
    node {
      echo "mkl"
    }
  },
  'Amalgamation': {
    node {
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
    }
  }
}
