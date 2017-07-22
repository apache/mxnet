#!/bin/bash
#
# This scripts installs the dependencies and compiles
# MXNet source.
#
# The script also installs the MXNet package for Python.
#

#set -ex

export MXNET_GITPATH="https://github.com/dmlc/mxnet.git"
if [ -z ${MXNET_TAG} ];
then
	#
	# TODO: Change this to latest tag 
	#       to avoid updating this value for every release
	#
	export MXNET_TAG="v0.10.0"
fi

export TARIKH=`/bin/date +%Y-%m-%d-%H:%M:%S`
if [ -z ${MXNET_HOME} ];
then
	export MXNET_HOME="$HOME/mxnet"
fi
export MXNET_HOME_OLD="$HOME/mxnet_${TARIKH}"
export MXNET_LOG=${MXNET_HOME}/buildMXNet_mac.log
# Insert the Homebrew directory at the top of your PATH environment variable
export PATH=/usr/local/bin:/usr/local/sbin:$PATH
export SLEEP_TIME=2

export BREW_PKGS="pkg-config python opencv numpy homebrew/science/openblas"
export PIP_PKGS_ALL="graphviz jupyter cython" 
export PIP_PKGS_USER="requests opencv-python"

LINE="########################################################################"

echo $LINE
echo " "
echo "MXNet is a flexible, efficient and scalable library for Deep Learning."
echo " "
echo "This script installs MXNet on MacOS in \${MXNET_HOME}"
echo "If not set, the default value of \${MXNET_HOME} = ~/mxnet"
echo "The current value of \${MXNET_HOME} = ${MXNET_HOME}"
echo " "
echo "If this directory is already present, it is renamed to retain earlier contents."
echo "You may want to check and delete this directory if not required."
echo " "
echo "This script has been tested on: MacOS El Capitan and Sierra"
echo " "
echo "If you face any problems with this script, please let us know at:"
echo "    https://stackoverflow.com/questions/tagged/mxnet"
echo " "
echo "Typical run-time for this script is around 10 minutes."
echo "If your environment has never been setup for development (e.g. gcc), "
echo "it could take up to 30 minutes or longer."
echo " "
MACOS_VERSION=`/usr/bin/uname -r`
echo "Your macOS version is: $MACOS_VERSION"
echo " "
echo $LINE
echo " "

while true; do
	echo "NOTE: This script supports Homebrew package manager only."
	echo " "
	echo "      It will install/update brew and following dependent packages required for MXNet."
	echo "      Dependent brew packages: ${BREW_PKGS}"
	echo "      Dependent pip  packages: ${PIP_PKGS_ALL} ${PIP_PKGS_USER}"
	read -p "Do you want to continue? (yes/no): " yn
	echo " "
	case $yn in
		[Yy]* ) break;;
		[Nn]* ) exit;;
		* ) echo "Please answer yes or no.";;
	esac
done

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
		mv ${MXNET_HOME} ${MXNET_HOME_OLD}
		echo " "
		echo "Renamed directory ${MXNET_HOME} to ${MXNET_HOME_OLD}"
		echo "You may want to check and delete this directory if not required."
		echo " "
		sleep ${SLEEP_TIME}
	fi

	echo "MXNET GIT Path = ${MXNET_GITPATH}"
	#echo "MXNET Tag = ${MXNET_TAG}"
	#echo "You can set \$MXNET_TAG to the appropriate github repo tag"
	#echo "If not set, the default value used is the latest release"
	echo " "
	sleep ${SLEEP_TIME}

	runme git clone ${MXNET_GITPATH} ${MXNET_HOME} --recursive
	sleep ${SLEEP_TIME}
	cd ${MXNET_HOME}
	echo " "
	#echo "Checkout tag = ${MXNET_TAG}"
	#runme git checkout ${MXNET_TAG}
	#echo " "
	sleep ${SLEEP_TIME}
}

echo " "
echo "BEGIN: Check/Install/Update Homebrew"
BREW_PATH=`/usr/bin/which brew`
if [[ (-z ${BREW_PATH})  ||  (! -f ${BREW_PATH}) ]];
then
	yes '' | /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
	ret=$?
	if [[ ${ret} != 0 ]]; then
		echo " "
		echo "ERROR: Return value non-zero for: homebrew installation using ruby"
		echo " "
		exit 1
	fi
else
	runme brew update
fi
echo "END: Check/Install/Update Homebrew"
echo $LINE
echo " "

echo " "
echo "BEGIN: Install dependent brew packages for MXNet: ${BREW_PKGS}"

runme brew tap homebrew/science

for pkg in ${BREW_PKGS}
do
	runme brew_pkg_install ${pkg}
done

echo "END: Install dependent brew packages for MXNet: ${BREW_PKGS}"
echo $LINE
echo " "

echo "BEGIN: Install dependent pip packages for MXNet: ${PIP_PKGS_ALL} ${PIP_PKGS_USER}"
runme pip install --upgrade pip
for pkg in ${PIP_PKGS_ALL}
do
	runme pip install ${pkg}
done
for pkg in ${PIP_PKGS_USER}
do
	runme pip install --user ${pkg}
done
echo "END: Install dependent pip packages for MXNet: ${PIP_PKGS_ALL} ${PIP_PKGS_USER}"
echo $LINE
echo " "

echo "BEGIN: Download MXNet"
download_mxnet
echo "END: Download MXNet"
sleep ${SLEEP_TIME}
echo $LINE
echo " "

# Compile MXNet: It assumes MXNet source is in ${MXNET_HOME}
echo "BEGIN: Compile MXNet"
cd ${MXNET_HOME}
runme cp make/osx.mk ./config.mk.tmp

touch ./config.mk
# rm any old setting of USE_BLAS, if present in config file
egrep -v "^USE_BLAS" ./config.mk.tmp                   >> ./config.mk
# add the new setting of USE_BLAS to the config file
echo "USE_BLAS = openblas"                             >> ./config.mk
echo "ADD_CFLAGS += -I/usr/local/opt/openblas/include" >> ./config.mk
echo "ADD_LDFLAGS += -L/usr/local/opt/openblas/lib"    >> ./config.mk
echo "ADD_LDFLAGS += -L/usr/local/lib/graphviz/"       >> ./config.mk
echo " "

echo "NOTE: The following compile-time configurations are being used."
echo "      If you want to change any of them, edit the following file"
echo "      in another terminal window and then press enter to continue."
echo " "
echo "      ${MXNET_HOME}/config.mk"
echo " "
echo $LINE
# remove commented and blank lines
egrep -v "^#" ${MXNET_HOME}/config.mk   | egrep -v "^$"
echo $LINE
echo " "
read -p "Press enter to continue ..."
echo " "
echo "Running Make"
echo " "
runme make -j$(sysctl -n hw.ncpu)
echo "END: Compile MXNet"
sleep ${SLEEP_TIME}
echo $LINE
echo " "

echo "BEGIN: Install MXNet package for Python"
runme cd ${MXNET_HOME}/python
runme sudo python setup.py install
echo "END: Install MXNet package for Python"
sleep ${SLEEP_TIME}
echo $LINE
echo " "

echo "BEGIN: Test MXNet"
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
	echo " "
	echo "SUCCESS: MXNet test passed"
	echo "SUCCESS: MXNet is successfully installed and works fine!"
	export MXNET_VERSION=`echo "import mxnet as mx; print(mx.__version__)" | python`
	echo "SUCCESS: MXNet Version is: $MXNET_VERSION"
	echo "END: Test MXNet"
	echo ":-)"
	echo " "
	echo "FYI : You can fine-tune MXNet run-time behavior using environment variables described at:"
	echo "      http://mxnet.io/how_to/env_var.html"
	echo " "
	echo "NEXT: Try the MNIST tutorial at: http://mxnet.io/tutorials/python/mnist.html"
	echo "      Try other tutorials at   : http://mxnet.io/tutorials"
	echo " "
	echo $LINE
	echo " "
	exit 0
else
	echo " "
	echo "ERROR: MXNet test failed"
	echo "END: Test MXNet"
	echo " "
	echo ":-("
	exit 1
fi
