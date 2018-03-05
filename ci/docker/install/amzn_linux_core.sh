set -ex
pushd .
yum install -y git
yum install -y wget
yum install -y sudo
yum install -y re2c
yum groupinstall -y 'Development Tools'

# Ninja
git clone --recursive https://github.com/ninja-build/ninja.git
cd ninja
./configure.py --bootstrap
cp ninja /usr/local/bin
popd

# CMake
pushd .
git clone --recursive https://github.com/Kitware/CMake.git --branch v3.10.2
cd CMake
./bootstrap
make -j$(nproc)
make install
popd