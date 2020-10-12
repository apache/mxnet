#!/usr/bin/env bash
set -e

# Install nvcc and setup environment variable
prefix=$DEPS_PATH
cuda=$CUDA_VERSION
libcuda=$LIBCUDA_VERSION
libcudnn=$LIBCUDNN_VERSION

cuda_major=$(echo $cuda | tr '-' '.' | cut -d. -f1,2 | tr '.' '-')
libcuda_major=$(echo $libcuda | cut -d. -f1)
libcudnn_major=$(echo $libcudnn | cut -d. -f1)

os_name=$(cat /etc/*release | grep '^ID=' | sed 's/^.*=//g')
os_version=$(cat /etc/*release | grep VERSION_ID | sed 's/^.*"\([0-9]*\)\.\([0-9]*\)"/\1\2/g')
os_id="${os_name}${os_version}"
if [[ $cuda_major == "9-0" ]]; then
    os_id="ubuntu1604"
fi

if [[ ! -d $DEPS_PATH/usr/local/cuda-$(echo $cuda | tr '-' '.' | cut -d. -f1,2) ]]; then
    # list of debs to download from nvidia

    files=( \
      "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/cuda-core-${cuda_major}_${cuda}_amd64.deb" \
      "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/cuda-cublas-${cuda_major}_${cuda}_amd64.deb" \
      "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/cuda-cublas-dev-${cuda_major}_${cuda}_amd64.deb" \
      "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/cuda-cudart-${cuda_major}_${cuda}_amd64.deb" \
      "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/cuda-cudart-dev-${cuda_major}_${cuda}_amd64.deb" \
      "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/cuda-curand-${cuda_major}_${cuda}_amd64.deb" \
      "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/cuda-curand-dev-${cuda_major}_${cuda}_amd64.deb" \
      "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/cuda-cufft-${cuda_major}_${cuda}_amd64.deb" \
      "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/cuda-cufft-dev-${cuda_major}_${cuda}_amd64.deb" \
      "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/cuda-nvrtc-${cuda_major}_${cuda}_amd64.deb" \
      "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/cuda-nvrtc-dev-${cuda_major}_${cuda}_amd64.deb" \
      "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/cuda-cusolver-${cuda_major}_${cuda}_amd64.deb" \
      "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/cuda-cusolver-dev-${cuda_major}_${cuda}_amd64.deb" \
      "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/cuda-misc-headers-${cuda_major}_${cuda}_amd64.deb" \
      "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/libcuda1-${libcuda_major}_${libcuda}_amd64.deb" \
      "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/nvidia-${libcuda_major}_${libcuda}_amd64.deb" \
      "http://developer.download.nvidia.com/compute/machine-learning/repos/${os_id}/x86_64/libcudnn${libcudnn_major}-dev_${libcudnn}_amd64.deb" \
    )

    for item in ${files[*]}
    do
        echo "Installing $item"
        curl -sL ${item} -o package.deb
        dpkg -X package.deb ${prefix}
        rm package.deb
    done

    cp ${prefix}/usr/include/x86_64-linux-gnu/cudnn_v${libcudnn_major}.h ${prefix}/include/cudnn.h
    ln -s libcudnn_static_v${libcudnn_major}.a ${prefix}/usr/lib/x86_64-linux-gnu/libcudnn.a
    cp ${prefix}/usr/local/cuda-$(echo $cuda | tr '-' '.' | cut -d. -f1,2)/lib64/*.a ${prefix}/lib/
fi

# @szha: this is a workaround for travis-ci#6522
set +e
