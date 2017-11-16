#!/usr/bin/env bash

set -e

wget https://julialang.s3.amazonaws.com/bin/linux/x64/0.5/julia-0.5.0-linux-x86_64.tar.gz
mv julia-0.5.0-linux-x86_64.tar.gz /tmp/
tar xfvz /tmp/julia-0.5.0-linux-x86_64.tar.gz
rm -f /tmp/julia-0.5.0-linux-x86_64.tar.gz

# tar extracted in current directory
ln -s -f ${PWD}/julia-3c9d75391c/bin/julia /usr/bin/julia
