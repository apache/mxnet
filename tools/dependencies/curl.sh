#!/usr/bin/env bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
