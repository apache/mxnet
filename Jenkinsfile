stage("Sanity Check") {
  node('master') {
    checkout scm
    sh 'git submodule update --init'
    sh 'tests/ci_build/ci_build.sh lint make cpplint'
    sh 'tests/ci_build/ci_build.sh lint make pylint'
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
