#!/bin/bash
#
# This scripts installs the dependencies and compiles
# MXNet source.
#
# The script also installs the MXNet package for Python.
#

#set -ex

export TARIKH=`/bin/date +%Y-%m-%d-%H:%M:%S`
export MXNET_HOME="$HOME/mxnet"
export MXNET_HOME_OLD="$HOME/mxnet_${TARIKH}"
export MXNET_LOG=${MXNET_HOME}/buildMXNet_mac.log
# Insert the Homebrew directory at the top of your PATH environment variable
export PATH=/usr/local/bin:/usr/local/sbin:$PATH
LINE="########################################################################"

echo $LINE
echo " "
echo "This script installs MXNet on MacOS in ${MXNET_HOME}"
echo "If this directory is already present, it is renamed to ${MXNET_HOME_OLD}"
echo "It has been tested to work successfully on MacOS El Capitan and Sierra"
echo "and is expected to work fine on other versions as well."
echo " "
echo "Approximate run-time is around 5 minutes."
echo " "
echo $LINE
sleep 2

#
# Install dependencies for MXNet
#

# Install Homebrew
yes '' | /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

brew_pkg_install () {
	pkg=$1
	brew ls --versions $pkg > /dev/null
	ret=$?
	if [[ ${ret} != 0 ]]; then
		echo "brew install $pkg"
		brew install $pkg
	else
		echo "$pkg already installed"
	fi
}

runme() {
	cmd=$*
	echo "$cmd"
	$cmd
	ret=$?
	if [[ ${ret} != 0 ]]; then
		echo " "
		echo "ERROR: Return value non-zero for: $cmd"
		echo " "
		exit 1
	fi
}

download_mxnet() {
	if [ -d ${MXNET_HOME} ]; then
		echo "Renaming directory ${MXNET_HOME} to ${MXNET_HOME_OLD}"
		mv ${MXNET_HOME} ${MXNET_HOME_OLD}
	fi
	echo "Downloading MXNET source repositories from github"
	git clone https://github.com/dmlc/mxnet.git ${MXNET_HOME} --recursive 
}

download_mxnet
runme brew update
runme brew_pkg_install pkg-config
runme brew_pkg_install python
brew install homebrew/science/openblas
runme brew_pkg_install opencv
# Needed for /usr/local/lib/graphviz to be created
runme brew_pkg_install graphviz
runme brew_pkg_install numpy

runme brew tap homebrew/science

runme pip install graphviz
runme pip install jupyter
runme pip install cython

#
# Compile MXNet. It assumes you have checked out MXNet source to ~/mxnet
#

cd ${MXNET_HOME}
runme cp make/osx.mk ./config.mk
runme echo "USE_BLAS = openblas" >> ./config.mk
runme echo "ADD_CFLAGS += -I/usr/local/opt/openblas/include" >> ./config.mk
runme echo "ADD_LDFLAGS += -L/usr/local/opt/openblas/lib" >> ./config.mk
runme echo "ADD_LDFLAGS += -L/usr/local/lib/graphviz/" >> ./config.mk
echo " "
echo "Running Make"
echo " "
runme make -j$(sysctl -n hw.ncpu)

#
# Install MXNet package for Python
#
echo "Installing MXNet package for Python..."
runme cd ${MXNET_HOME}/python
runme sudo python setup.py install

#
# Test MXNet
#
echo "Testing MXNet now..."
python  << END > mxnet_test.log
import mxnet as mx
a = mx.nd.ones((2, 3));
print ((a*2).asnumpy());
END
cat << END > mxnet_test.expected
[[ 2.  2.  2.]
 [ 2.  2.  2.]]
END
diff mxnet_test.log mxnet_test.expected
if [[ $? = 0 ]]; then
	echo $LINE
	echo " "
	echo "SUCCESS: MXNet test passed"
	echo "SUCCESS: MXNet is successfully installed and works fine!"
	echo ":-)" | banner -w 40
	echo " "
	echo $LINE
	exit 0
else
	echo $LINE
	echo " "
	echo "ERROR: MXNet test failed"
	echo ":-(" | banner -w 40
	echo " "
	echo $LINE
	exit 1
fi
