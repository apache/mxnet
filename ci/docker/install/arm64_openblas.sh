set -ex
pushd .
wget -nv https://api.github.com/repos/xianyi/OpenBLAS/git/refs/heads/master -O openblas_version.json
echo "Using openblas:"
cat openblas_version.json
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make -j$(nproc) TARGET=ARMV8
make install
ln -s /opt/OpenBLAS/lib/libopenblas.so /usr/lib/libopenblas.so
ln -s /opt/OpenBLAS/lib/libopenblas.a /usr/lib/libopenblas.a
ln -s /opt/OpenBLAS/lib/libopenblas.a /usr/lib/liblapack.a
popd