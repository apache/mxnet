#!/usr/bin/env bash

# This script builds the static library of openssl that can be used as dependency of mxnet.
OPENSSL_VERSION=1.0.2l
if [[ ! -f $DEPS_PATH/lib/libssl.a ]] || [[ ! -f $DEPS_PATH/lib/libcrypto.a ]]; then
    # download and build openssl
    >&2 echo "Building openssl..."
    OPENSSL_VERSION=$(echo $OPENSSL_VERSION | sed 's/\./_/g')
    curl -s -L https://github.com/openssl/openssl/archive/OpenSSL_$OPENSSL_VERSION.zip -o $DEPS_PATH/openssl.zip
    unzip -q $DEPS_PATH/openssl.zip -d $DEPS_PATH
    cd $DEPS_PATH/openssl-OpenSSL_$OPENSSL_VERSION
    if [[ $PLATFORM == 'linux' ]]; then
        TARGET=linux-x86_64
    elif [[ $PLATFORM == 'darwin' ]]; then
        TARGET=darwin64-x86_64-cc
    fi
    ./Configure no-shared no-zlib --prefix=$DEPS_PATH --openssldir=$DEPS_PATH/ssl $TARGET
    make
    make install
    cd -
fi
