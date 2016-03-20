data_path="./data"
if [ ! -d "$data_path" ]; then
  mkdir -p "$data_path"
fi

mnist_data_path="./data/mnist.zip"
if [ ! -f "$mnist_data_path" ]; then
  wget http://webdocs.cs.ualberta.ca/~bx3/data/mnist.zip -P $data_path
  cd $data_path
  unzip -u mnist.zip
fi
