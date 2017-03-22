if [ ! -f "./mnist.zip" ]; then
  wget http://webdocs.cs.ualberta.ca/~bx3/data/mnist.zip
  unzip -u mnist.zip
fi
make lenet_with_mxdataiter
LD_LIBRARY_PATH=../lib/linux ./lenet_with_mxdataiter
