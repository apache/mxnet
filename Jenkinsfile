stage('Build') {
  node {
    echo "build"
  }
}

stage('Test') {
  parallel linux: {
    node {
      echo "test"
    }
  },
  windows: {
    node {
      echo "test"
    }
  }
}
