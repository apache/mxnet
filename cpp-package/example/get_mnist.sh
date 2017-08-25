if [ ! -d "./mnist_data" ]; then
  mkdir mnist_data
  (cd mnist_data; wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
  (cd mnist_data; wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)
  (cd mnist_data; wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz)
  (cd mnist_data; wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)
  (cd mnist_data; gzip -d *.gz)
fi
echo "Data downloaded"
