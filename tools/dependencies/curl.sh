#!/usr/bin/env bash

# This script builds the static library of libcurl that can be used as dependency of mxnet.
LIBCURL_VERSION=7.61.0
if [[ ! -f $DEPS_PATH/lib/libcurl.a ]]; then
    # download and build libcurl
    >&2 echo "Building libcurl..."
    curl -s -L https://curl.haxx.se/download/curl-$LIBCURL_VERSION.zip -o $DEPS_PATH/libcurl.zip
    unzip -q $DEPS_PATH/libcurl.zip -d $DEPS_PATH
    cd $DEPS_PATH/curl-$LIBCURL_VERSION
    if [[ $PLATFORM == 'linux' ]]; then
        CONFIG_FLAG=""
    elif [[ $PLATFORM == 'darwin' ]]; then
        CONFIG_FLAG="--with-darwinssl"
    fi
    ./configure $CONFIG_FLAG \
                --with-zlib \
                --with-nghttps2 \
                --without-zsh-functions-dir \
                --without-librtmp \
                --without-libssh2 \
                --disable-debug \
                --disable-curldebug \
                --enable-symbol-hiding=yes \
                --enable-optimize=yes \
                --enable-shared=no \
                --enable-http=yes \
                --enable-ipv6=yes \
                --disable-ftp \
                --disable-ldap \
                --disable-ldaps \
                --disable-rtsp \
                --disable-proxy \
                --disable-dict \
                --disable-telnet \
                --disable-tftp \
                --disable-pop3 \
                --disable-imap \
                --disable-smb \
                --disable-smtp \
                --disable-gopher \
                --disable-manual \
                --prefix=$DEPS_PATH
    make
    make install
    cd -
fi
