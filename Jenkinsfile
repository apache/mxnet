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
