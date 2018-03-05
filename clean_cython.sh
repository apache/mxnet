#!/bin/sh
for i in $(find ./python/mxnet -type f -name "*.so"); do rm -f $i; done
for i in $(find ./python/mxnet -type d -name "cython_debug"); do rm -rf $i; done
