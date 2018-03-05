set -ex
yum groupinstall -y "Development Tools"
yum install -y mlocate python27 python27-setuptools python27-tools python27-numpy python27-scipy python27-nose python27-matplotlib unzip
ln -s -f /usr/bin/python2.7 /usr/bin/python2
wget -nv https://bootstrap.pypa.io/get-pip.py
python2 get-pip.py
$(which easy_install-2.7) --upgrade pip
if [ -f /usr/local/bin/pip ] && [ -f /usr/bin/pip ]; then
    mv /usr/bin/pip /usr/bin/pip.bak
    ln /usr/local/bin/pip /usr/bin/pip
fi

ln -s -f /usr/local/bin/pip /usr/bin/pip
for i in ipython[all] jupyter pandas scikit-image h5py pandas sklearn sympy; do echo "${i}..."; pip install -U $i >/dev/null; done


set -ex
pushd .
wget -nv https://bootstrap.pypa.io/get-pip.py
mkdir py3
cd py3
wget -nv https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz
tar -xvzf Python-3.5.2.tgz
cd Python-3.5.2
yum install -y zlib-devel openssl-devel sqlite-devel bzip2-devel gdbm-devel ncurses-devel xz-devel readline-devel
./configure --prefix=/opt/ --with-zlib-dir=/usr/lib64
make -j$(nproc)
mkdir /opt/bin
mkdir /opt/lib
make install
ln -s -f /opt/bin/python3 /usr/bin/python3
cd ../..
python3 get-pip.py
ln -s -f /opt/bin/pip /usr/bin/pip3

mkdir -p ~/.local/lib/python3.5/site-packages/
pip3 install numpy
popd