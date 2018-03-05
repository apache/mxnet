set -ex
pushd .
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make -j$(nproc) TARGET=ARMV8 ARM_SOFTFP_ABI=1 HOSTCC=gcc NOFORTRAN=1 libs
cp libopenblas.a /usr/local/lib
popd