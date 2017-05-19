cd $(dirname $(readlink -f $0))/../example && \
  ln -s ../../lib/libmxnet.so . &&
  cp ../../build/cpp-package/example/test_score . && \
  ./get_mnist.sh && \
  ./test_score 0.94 && \
  rm libmxnet.so
