#!/usr/bin/env bash
make_config=config/pip_${PLATFORM}_${VARIANT}.mk
if [[ ! -f $make_config ]]; then
    >&2 echo "Couldn't find make config $make_config for the current settings."
    exit 1
fi

# Set up path as temporary working directory
mkdir -p $DEPS_PATH

# Set up shared dependencies:
if [[ $DEBUG -eq 1 ]]; then
    scripts/make_shared_dependencies.sh
    scripts/make_ps_dependencies.sh
    if [[ ! $PLATFORM == 'darwin' ]]; then
        scripts/make_openblas.sh
    fi
else
    scripts/make_shared_dependencies.sh > /dev/null
    scripts/make_ps_dependencies.sh > /dev/null
    if [[ ! $PLATFORM == 'darwin' ]]; then
        scripts/make_openblas.sh > /dev/null
    fi
fi
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(dirname $(find $DEPS_PATH -type f -name 'libprotoc*' | grep protobuf | head -n 1)):$DEPS_PATH/lib

# Although .so/.dylib building is explicitly turned off for most libraries, sometimes
# they still get created. So, remove them just to make sure they don't
# interfere, or otherwise we might get libmxnet.so that is not self-contained.
# For CUDA, since we cannot redistribute the shared objects or perform static linking,
# we DO want to keep the shared objects around, hence performing deletion before cuda setup.
find deps/{lib,lib64} -maxdepth 1 -type f -name '*.so' -or -name '*.so.*' -or -name '*.dylib' | grep -v 'libproto' | xargs rm

if [[ $PLATFORM == 'linux' ]]; then

    if [[ $VARIANT == cu* ]]; then
        # download and install cuda and cudnn, and set paths
        scripts/setup_gpu_build_tools.sh
    fi
fi

