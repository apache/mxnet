#!/usr/bin/env bash
make_config=config/pip_${PLATFORM}_${VARIANT}.mk
if [[ ! -f $make_config ]]; then
    >&2 echo "Couldn't find make config $make_config for the current settings."
    exit 1
fi

# If a travis build is from a tag, use this tag for fetching the corresponding release
if [[ ! -z $TRAVIS_TAG ]]; then
    GIT_ADDITIONAL_FLAGS="-b $(echo $TRAVIS_TAG | sed 's/^patch-//g')"
fi

rm -rf mxnet-build
git clone --recursive https://github.com/dmlc/mxnet mxnet-build $GIT_ADDITIONAL_FLAGS

>&2 echo "Now building mxnet..."
cp $make_config mxnet-build/config.mk
cd mxnet-build

make -j $NUM_PROC DEPS_PATH=$DEPS_PATH || exit 1;
#if [[ ! $VARIANT == *mkl ]]; then
    # change RTLD_LOCAL to RTLD_GLOBAL for cython
    # szha@: this is a hack...
    # >&2 echo "Making cython since this variant is not MKL"
    #sed -i 's/RTLD_LOCAL/RTLD_GLOBAL/' python/mxnet/base.py
    #make cython || exit 1;
#fi
cd ../

if [[ $VARIANT == *mkl ]]; then
    >&2 echo "Copying MKL shared objects."
    if [[ ! $PLATFORM == 'darwin' ]]; then
        mv deps/lib/libmklml_intel.so mxnet-build/lib
        mv deps/lib/libmklml_gnu.so mxnet-build/lib
        mv deps/lib/libiomp5.so mxnet-build/lib
    else
        mv deps/lib/libmklml.dylib mxnet-build/lib
        mv deps/lib/libiomp5.dylib mxnet-build/lib
        install_name_tool -change '@rpath/libmklml.dylib' '@loader_path/libmklml.dylib' mxnet-build/lib/libmxnet.so
        install_name_tool -change '@rpath/libiomp5.dylib' '@loader_path/libiomp5.dylib' mxnet-build/lib/libmxnet.so
    fi
    mv deps/license.txt mxnet-build/MKLML_LICENSE
    rm deps/include/mkl*
fi

# Print the linked objects on libmxnet.so
>&2 echo "Checking linked objects on libmxnet.so..."
if [[ ! -z $(command -v readelf) ]]; then
    readelf -d mxnet-build/lib/libmxnet.so
elif [[ ! -z $(command -v otool) ]]; then
    otool -L mxnet-build/lib/libmxnet.so
else
    >&2 echo "Not available"
fi
