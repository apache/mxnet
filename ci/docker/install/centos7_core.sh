set -ex

# Multipackage installation does not fail in yum
yum -y install epel-release
yum -y install git
yum -y install wget
yum -y install atlas-devel # Provide clbas headerfiles
yum -y install openblas-devel
yum -y install lapack-devel
yum -y install opencv-devel
yum -y install openssl-devel
yum -y install gcc-c++
yum -y install make
yum -y install cmake
yum -y install wget
yum -y install unzip
yum -y install ninja-build