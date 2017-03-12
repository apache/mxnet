stage("Sanity Check") {
  parallel 'C++ Lint': {
    node('master') {
      sh 'tests/ci_build/ci_build.sh lint make cpplint'
    }
  },
  'Python Lint': {
    node('master') {
      sh 'tests/ci_build/ci_build.sh lint make pylint'
    }
  }
}
stage('Build') {
  node {
    echo "build"
  }
}

stage('Test') {
  parallel linux: {
    node {
      echo "test"
      sh "ls"
      sh "pwd"
    }
  },
  windows: {
    node {
      echo "test"
    }
  }
}
