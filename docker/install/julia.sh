#!/usr/bin/env bash
# install libraries for mxnet's julia package on ubuntu

# the julia version shipped with ubuntu (version 0.4) is too low. so download a
# new version
# apt-get install -y julia

wget -q https://julialang.s3.amazonaws.com/bin/linux/x64/0.5/julia-0.5.1-linux-x86_64.tar.gz
tar -zxf julia-0.5.1-linux-x86_64.tar.gz
rm julia-0.5.1-linux-x86_64.tar.gz
ln -s $(pwd)/julia-6445c82d00/bin/julia /usr/bin/julia
