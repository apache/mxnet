data_path="./data"
if [ ! -d "$data_path" ]; then
  mkdir -p "$data_path"
fi

cifar_data_path="./data/cifar10.zip"
if [ ! -f "$cifar_data_path" ]; then
  wget http://webdocs.cs.ualberta.ca/~bx3/data/cifar10.zip -P $data_path
  cd $data_path
  unzip -u cifar10.zip
fi