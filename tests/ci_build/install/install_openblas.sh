#!/usr/bin/env bash

set -e

git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make FC=gfortran -j $(($(nproc) + 1))
make PREFIX=/usr/local install
