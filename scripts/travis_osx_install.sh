#!/bin/bash

brew update
brew tap homebrew/science
brew info opencv
brew install graphviz
brew install opencv

if [ ${TASK} == "python-package3" ]; then
    wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
else
    wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda-latest-MacOSX-x86_64.sh
fi


bash conda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
# Useful for debugging any issues with conda
conda info -a

if [ ${TASK} == "package3" ]; then
    conda create -n myenv python=3.4
else
    conda create -n myenv python=2.7
fi
source activate myenv
conda install numpy scipy matplotlib nose
python -m pip install graphviz