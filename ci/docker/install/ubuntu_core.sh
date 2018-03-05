set -ex
apt-get update
apt-get install -y \
    build-essential \
    git \
    libopenblas-dev \
    liblapack-dev \
    libopencv-dev \
    libcurl4-openssl-dev \
    cmake \
    wget \
    unzip \
    sudo \
    software-properties-common \
    ninja-build \
    python-pip

# Link Openblas to Cblas as this link does not exist on ubuntu16.04
ln -s /usr/lib/libopenblas.so /usr/lib/libcblas.so
pip install cpplint==1.3.0 pylint==1.8.2