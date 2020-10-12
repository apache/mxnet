#!/usr/bin/env bash

# Dependencies that are shared by variants are:
PROTOBUF_VERSION=2.5.0
ZEROMQ_VERSION=4.2.2
LZ4_VERSION=r130
CITYHASH_VERSION=1.1.1

if [[ $PLATFORM == 'darwin' ]]; then
    DY_EXT="dylib"
else
    DY_EXT="so"
fi

# Set up ps-lite dependencies:

LIBPROTOBUF="$DEPS_PATH/lib/libprotobuf.$DY_EXT"
LIBPROTOC="$DEPS_PATH/lib/libprotoc.$DY_EXT"
if [[ ! -e $LIBPROTOBUF ]] || [[ ! -e $LIBPROTOC ]]; then
    # Download and build protobuf
    >&2 echo "Building protobuf..."
    curl -s -L https://github.com/google/protobuf/releases/download/v$PROTOBUF_VERSION/protobuf-$PROTOBUF_VERSION.zip -o $DEPS_PATH/protobuf.zip
    unzip -q $DEPS_PATH/protobuf.zip -d $DEPS_PATH
    cd $DEPS_PATH/protobuf-$PROTOBUF_VERSION
    ./configure -prefix=$DEPS_PATH
    make --quiet -j $NUM_PROC || exit 1;
    make install;
    cd -;
fi

if [[ ! -f $DEPS_PATH/lib/libcityhash.a ]]; then
    # Download and build cityhash
    >&2 echo "Building cityhash..."
    git clone https://github.com/google/cityhash $DEPS_PATH/cityhash-$CITYHASH_VERSION
    cd $DEPS_PATH/cityhash-$CITYHASH_VERSION
    git reset --hard 8af9b8c2b889d80c22d6bc26ba0df1afb79a30db
    ./configure -prefix=$DEPS_PATH --enable-sse4.2
    make --quiet -j $NUM_PROC CXXFLAGS="-g -O3 -msse4.2" || exit 1;
    make install;
    cd -;
fi

if [[ ! -f $DEPS_PATH/lib/libzmq.a ]]; then
    # Download and build zmq
    >&2 echo "Building zmq..."
    curl -s -L https://github.com/zeromq/libzmq/archive/v$ZEROMQ_VERSION.zip -o $DEPS_PATH/zeromq.zip
    unzip -q $DEPS_PATH/zeromq.zip -d $DEPS_PATH
    mkdir $DEPS_PATH/libzmq-$ZEROMQ_VERSION/build
    cd $DEPS_PATH/libzmq-$ZEROMQ_VERSION/build
    cmake -q \
          -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=$DEPS_PATH \
          -D WITH_LIBSODIUM=OFF \
          -D BUILD_SHARED_LIBS=OFF .. || exit 1;
    make --quiet -j $NUM_PROC || exit 1;
    make install;
    cd -;
fi

if [[ ! -f $DEPS_PATH/lib/liblz4.a ]]; then
    # Download and build lz4
    >&2 echo "Building lz4..."
    curl -s -L https://github.com/lz4/lz4/archive/$LZ4_VERSION.zip -o $DEPS_PATH/lz4.zip
    unzip -q $DEPS_PATH/lz4.zip -d $DEPS_PATH
    cd $DEPS_PATH/lz4-$LZ4_VERSION
    make --quiet -j $NUM_PROC || exit 1;
    PREFIX=$DEPS_PATH make install;
    cd -;
fi
