stage("Sanity Check") {
  parallel 'C++ Lint': {
    node {
      sh 'tests/ci_build/ci_build.sh cpu make cpplint'
    }
  },
  'Python Lint': {
    node {
      sh 'tests/ci_build/ci_build.sh cpu make pylint'
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
