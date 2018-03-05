set -ex
pushd .
yum install -y python27 python27-setuptools
git clone https://github.com/opencv/opencv
cd opencv
mkdir -p build
cd build
cmake -DBUILD_opencv_gpu=OFF -DWITH_EIGEN=ON -DWITH_TBB=ON -DWITH_CUDA=OFF -DWITH_1394=OFF \
-DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -GNinja ..
ninja install
popd