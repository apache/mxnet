cd $(dirname $(readlink -f $0))/../example && \
  ln -sf ../../lib/libmxnet.so . &&
  cp ../../build/cpp-package/example/test_score . && \
  ./get_mnist.sh && \
  ./test_score 0.93
