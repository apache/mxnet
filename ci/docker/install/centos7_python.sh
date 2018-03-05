set -ex

 # Python 2.7 is installed by default, install 3.6 on top
yum -y install https://centos7.iuscommunity.org/ius-release.rpm
yum -y install python36u

# Install PIP
curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
python2.7 get-pip.py
python3.6 get-pip.py

pip2 install nose pylint numpy nose-timer requests h5py scipy
pip3 install nose pylint numpy nose-timer requests h5py scipy