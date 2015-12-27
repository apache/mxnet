if [ ! -d "./data" ]; then
  mkdir -p ./data
fi
if [ ! -d "./data/mnist.zip" ]; then
    wget http://webdocs.cs.ualberta.ca/~bx3/data/mnist.zip -P data/
fi
cd data
rm *-ubyte
unzip -u mnist.zip