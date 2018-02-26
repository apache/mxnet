# If you need to build <= v0.12.0 then use a Python 2 environment
# mxdoc.py - a sphinx extension, was not Python 3 compatible in the old versions
# source activate mxnet_p27

# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    doxygen \
    software-properties-common

pip install --user \
    beautifulsoup4 \
    breathe \
    CommonMark==0.5.4 \
    h5py \
    mock==1.0.1 \
    pypandoc \
    recommonmark==0.4.0 \
    sphinx==1.5.6 
    
# Recommonmark/Sphinx errors: https://github.com/sphinx-doc/sphinx/issues/3800


# Setup scala
echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
sudo apt-get update
sudo apt-get install -y \
  sbt \
  scala

# Cleanup
sudo apt autoremove -y

# Make docs using the manual way
# cd .. && make html USE_OPENMP=0
# using the docker way
# sudo make docs 

