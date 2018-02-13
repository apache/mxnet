#!/bin/bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#
# This scripts installs the dependencies and compiles
# MXNet source.
#
# The script also installs the MXNet package for Python.
#

#set -ex

export MXNET_GITPATH="https://github.com/apache/incubator-mxnet"


if [ -z ${MXNET_TAG} ];
then
    export MXNET_BRANCH_TAG=""
else
    export MXNET_BRANCH_TAG="--branch $MXNET_TAG"
fi

export TARIKH=`/bin/date +%Y-%m-%d-%H:%M:%S`
if [ -z ${MXNET_HOME} ];
then
	export MXNET_HOME="$HOME/mxnet"
fi
export MXNET_HOME_OLD="$HOME/mxnet_${TARIKH}"
export MXNET_LOG=${MXNET_HOME}/buildMXNet_mac.log

# Insert the Homebrew directory at the top of your PATH environment variable
export PATH="$PATH:/usr/local/bin:/usr/local/sbin" # for brew
export PATH="$PATH:/usr/bin:/opt/local/bin"        # for macports

export MACPORTS_WEB="https://guide.macports.org/chunked/installing.macports.html"

export BREW_PKGS="pkg-config python   opencv graphviz homebrew/science/openblas"
export PORT_PKGS="pkgconfig  python36 opencv graphviz openblas-devel"

# graphviz, opencv-python skipped since already installed via brew/port
export PIP_PKGS_ALL="cython numpy"
export PIP_PKGS_USER="requests jupyter"

export SLEEP_TIME=2
LINE="########################################################################"

print_intro_msg() {
	#
	# NOTE: Please test and ensure that the message does NOT scroll
	#       beyond the standard 80x25 format of a terminal shell.
	#
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
	echo "This script has been tested on: MacOS El Capitan (10.11) and Sierra (10.12)"
	echo " "
	echo "If you face any problems with this script, please let us know at:"
	echo "    https://discuss.mxnet.io/"
	echo " "
	echo "Typical run-time for this script is around 10 minutes."
	echo "If your environment has never been setup for development (e.g. gcc), "
	echo "it could take up to 30 minutes or longer."
	echo " "
	MACOS_VERSION=`/usr/bin/uname -r`
	echo "Your macOS version is: $MACOS_VERSION"
	echo " "
	echo $LINE
	read -p "Do you want to continue? (y/n): " response
	echo " "
	while true; do
		case $response in
			[Yy]* ) break;;
			[Nn]* ) exit;;
			* ) echo "Please answer yes or no.";;
		esac
	done
	echo " "
	echo " "
	echo "MXNET GIT Path = ${MXNET_GITPATH}"
	echo "MXNET Tag = ${MXNET_TAG}"
	echo "You can set \$MXNET_TAG to the appropriate github repo tag"
	echo "If not set, the default value used is the latest version on master"
	read -p "Do you want to get a list of available tags? (y/n): " response
	while true; do
		case $response in
			[Yy]* ) 
				echo "Available tags are:"
				git ls-remote --tags ${MXNET_GITPATH} | sed 's/refs\/tags\///' | grep -v v | grep -v 201 \
				    | grep -v "{}" | awk '{ print "   ", $2 }'; 
				break;;
			[Nn]* ) break;;
			* ) echo "Please answer yes or no.";;
		esac
	done
	read -p "Do you want to continue? (y/n): " response
	echo " "
	while true; do
		case $response in
			[Yy]* ) break;;
			[Nn]* ) exit;;
			* ) echo "Please answer yes or no.";;
		esac
	done
} # print_intro_msg()

# wrapper routine to stop the script if the command invoked returns error
chkret() {
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
} # chkret()

chk_mac_vers() {
	export mac_vers=`sw_vers -productVersion | cut -d '.' -f 1,2`
	if [[ $mac_vers != "10.11" && $mac_vers != "10.12" ]];
	then
		echo " "
		echo "ERROR: macOS version $mac_vers NOT supported."
		echo " "
		echo "Your macOS version is:"
		sw_vers
		echo " "
		exit 1
	fi
} # chk_mac_vers()

install_brew() {
	echo " "
	while true; do
		echo "This script will install/update brew and "
		echo "following dependent packages required for MXNet."
		echo "      Dependent brew packages: ${BREW_PKGS}"
		echo "      Dependent pip  packages: ${PIP_PKGS_ALL} ${PIP_PKGS_USER}"
		read -p "Do you want to continue? (y/n): " response
		echo " "
		case $response in
			[Yy]* ) break;;
			[Nn]* ) exit;;
			* ) echo "Please answer yes or no.";;
		esac
	done

	echo " "
	echo "BEGIN: Check/Install/Update Homebrew"
	BREW_PATH=`which brew`
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
		chkret brew update
	fi
	echo "END: Check/Install/Update Homebrew"
	echo $LINE
	echo " "

	echo "BEGIN: Install dependent brew packages for MXNet: ${BREW_PKGS}"

	chkret brew tap homebrew/science

	# install each individually to see progress for each
	for pkg in ${BREW_PKGS}
	do
		chkret brew_pkg_install ${pkg}
	done

	echo "END: Install dependent brew packages for MXNet: ${BREW_PKGS}"
	echo $LINE
	echo " "
} # install_brew()

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
} # brew_pkg_install

install_port () {
	echo " "
	while true; do
		echo "This script will install/update port and "
		echo "following dependent packages required for MXNet."
		echo "      Dependent port packages: ${PORT_PKGS}"
		echo "      Dependent pip  packages: ${PIP_PKGS_ALL} ${PIP_PKGS_USER}"
		read -p "Do you want to continue? (y/n): " response
		echo " "
		case $response in
			[Yy]* ) break;;
			[Nn]* ) exit;;
			* ) echo "Please answer yes or no.";;
		esac
	done

	echo " "
	echo "BEGIN: Check/Install/Update port"
	MACPORTS_PATH=`which port`
	if [[ (-z ${MACPORTS_PATH})  ||  (! -f ${MACPORTS_PATH}) ]];
	then
		echo " "
		echo "ERROR: Please install port for your macOS version from:"
		echo " "
		echo $MACPORTS_WEB
		echo " "
		exit 1
	else
		echo "NOTE: Updating port if required"
		export SLEEP_TIME=2
		sudo port upgrade outdated
		echo " "
		echo "port version is:"
		port version
		echo " "
	fi
	echo "END: Check/Install/Update port"
	echo $LINE
	echo " "

	echo "BEGIN: Install dependent port packages for MXNet: ${PORT_PKGS}"
	echo " "
	#sudo port install python36-readline
	# install each individually to see progress for each
	for pkg in ${PORT_PKGS}
	do
		chkret sudo port install ${pkg}
	done
	if [[ ! -f /opt/local/include/cblas.h ]];
	then
		sudo ln -s /opt/local/include/cblas_openblas.h /opt/local/include/cblas.h
	fi
	#if [[ ! -f /usr/local/opt/openblas/lib/libopenblas.a ]];
	#then
	#	sudo mkdir -p /usr/local/opt/openblas/lib
	#	sudo ln -s /opt/local/lib/libopenblas.a /usr/local/opt/openblas/lib/libopenblas.a
	#fi

	echo " "
	echo "END: Install dependent port packages for MXNet: ${PORT_PKGS}"
	echo $LINE
	echo " "
} # install_port

install_mac_pkg_manager() {
	BREW_PATH=`which brew`
	if [[ (-z ${BREW_PATH})  ||  (! -f ${BREW_PATH}) ]];
	then
		echo "NOTE: brew NOT installed"
		export MAC_BREW=0
	else
		echo "NOTE: brew installed"
		export MAC_BREW=1
		export PKG_MGR="brew"
	fi

	MACPORTS_PATH=`which port`
	if [[ (-z ${MACPORTS_PATH})  ||  (! -f ${MACPORTS_PATH}) ]];
	then
		echo "NOTE: port NOT installed"
		export MAC_PORT=0
	else
		echo "NOTE: port installed"
		export MAC_PORT=1
		export PKG_MGR="port"
	fi

	if [[ $MAC_PORT -eq 1 && $MAC_BREW -eq 1 ]];
	then
		echo "NOTE: Both port and brew installed"
		export MAC_PKG_ASK=1
		export PKG_MGR=""
	elif [[ $MAC_PORT -eq 0 && $MAC_BREW -eq 0 ]];
	then
		echo "NOTE: Neither port and brew installed"
		export MAC_PKG_ASK=1
		export PKG_MGR=""
	else
		export MAC_PKG_ASK=0

		while true; do
			echo "NOTE: Using the already installed package manager: $PKG_MGR"
			read -p "Do you want to continue? (y/n): " response
			echo " "
			case $response in
				[Yy]* ) break;;
				[Nn]* ) exit;;
				* ) echo "Please answer yes or no.";;
			esac
		done
	fi

	if [[ $MAC_PKG_ASK -eq 1 ]];
	then
		export MAC_BREW=0
		export MAC_PORT=0
		while true; do
			echo " "
			echo "NOTE: This script supports Homebrew OR Port package manager."
			echo " "
			read -p "Which package manager do you want to use? (b/p): " pkg_mgr
			echo " "
			case $pkg_mgr in
				[Bb]* ) export MAC_BREW=1; break;;
				[Pp]* ) export MAC_PORT=1; break;;
				* ) echo "Please answer: b or p";;
			esac
		done
	fi

	if [[ $MAC_PORT -eq 1 ]];
	then
		install_port
	else
		install_brew
	fi
} # install_mac_pkg_manager

install_dep_pip_for_mxnet() {
	echo " "
	echo "BEGIN: Install dependent pip packages for MXNet: "
	echo "${PIP_PKGS_ALL} ${PIP_PKGS_USER}"
	echo " "

	# NOTE: sudo used here
	chkret sudo easy_install pip
	chkret sudo pip install --upgrade pip
	for pkg in ${PIP_PKGS_ALL}
	do
		chkret sudo pip install ${pkg}
	done
	#chkret sudo pip install --upgrade numpy

	# NOTE: no sudo used here
	for pkg in ${PIP_PKGS_USER}
	do
		chkret pip install --user ${pkg}
	done

	echo "END: Install dependent pip packages for MXNet: ${PIP_PKGS_ALL} ${PIP_PKGS_USER}"
	echo $LINE
	echo " "
} # install_dep_pip_for_mxnet()

# check if mxnet is already installed through other means
chk_mxnet_installed() {
	mxnet_installed=`pip list --format=columns | grep mxnet`
	if [ "$mxnet_installed" != "" ]
	then
		mxnet_version=`echo $mxnet_installed | awk '{print $2}'`
		echo "MXNet ${mxnet_version} is already installed."
		echo "This installation might interfere with current installation attempt."
		read -p "Do you want to remove installed version? (y/n): " response
		while true; do
			case $response in
            	[Yy]* ) 
					sudo -H pip uninstall mxnet
					chk_mxnet_installed
					break
					;;
            	[Nn]* ) 
					while true; do
						read -p "Do you want to continue? (y/n): " response1
        				echo " "
        				case $response1 in
            				[Yy]* ) break 2;; # break out of nested loop
            				[Nn]* ) exit;;
            				* ) echo "Please answer yes or no.";;
        				esac
					done
					;;
            	* ) echo "Please answer yes or no.";;
        	esac
		done
	fi
} # chk_mxnet

download_mxnet() {
	echo " "
	echo "BEGIN: Download MXNet"
	if [ -d ${MXNET_HOME} ]; then
		mv ${MXNET_HOME} ${MXNET_HOME_OLD}
		echo " "
		echo "Renamed directory ${MXNET_HOME} to ${MXNET_HOME_OLD}"
		echo "You may want to check and delete this directory if not required."
		echo " "
		sleep ${SLEEP_TIME}
	fi

	
	chkret git clone ${MXNET_BRANCH_TAG} ${MXNET_GITPATH}.git ${MXNET_HOME} --recursive
	sleep ${SLEEP_TIME}
	cd ${MXNET_HOME}
	echo " "
	sleep ${SLEEP_TIME}
	echo "END: Download MXNet"
	echo $LINE
	echo " "
} # download_mxnet

compile_mxnet() {
	# Compile MXNet: It assumes MXNet source is in ${MXNET_HOME}
	echo "BEGIN: Compile MXNet"
	cd ${MXNET_HOME}
	chkret cp make/osx.mk ./config.mk.tmp

	touch ./config.mk
	# rm any old setting of USE_BLAS, if present in config file
	egrep -v "^USE_BLAS" ./config.mk.tmp                   >> ./config.mk
	# add the new setting of USE_BLAS to the config file
	echo "USE_BLAS = openblas"                             >> ./config.mk

	if [[ $MAC_PORT -eq 1 ]];
	then
		echo "ADD_CFLAGS  += -I/opt/local/lib"            >> ./config.mk
		echo "ADD_LDFLAGS += -L/opt/local/lib"            >> ./config.mk
		echo "ADD_LDFLAGS += -L/opt/local/lib/graphviz/"  >> ./config.mk
	else
		echo "ADD_CFLAGS  += -I/usr/local/opt/openblas/include" >> ./config.mk
		echo "ADD_LDFLAGS += -L/usr/local/opt/openblas/lib"     >> ./config.mk
		echo "ADD_LDFLAGS += -L/usr/local/lib/graphviz/"        >> ./config.mk
	fi
	echo " "

	echo "NOTE: The following compile-time configurations will be used."
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
	chkret make -j$(sysctl -n hw.ncpu)
	echo "END: Compile MXNet"
	sleep ${SLEEP_TIME}
	echo $LINE
	echo " "
} # compile_mxnet

install_mxnet_python() {
	echo " "
	echo "BEGIN: Install MXNet package for Python"
	chkret cd ${MXNET_HOME}/python
	chkret sudo python setup.py install
	echo "END: Install MXNet package for Python"
	sleep ${SLEEP_TIME}
	echo $LINE
	echo " "
} # install_mxnet_python


test_mxnet_python() {
	echo "BEGIN: Test MXNet"
	rm -f mxnet_test.log
	python  << END > mxnet_test.log
import mxnet as mx
a = mx.nd.ones((2, 3));
print ((a*2).asnumpy());
END
	rm -f mxnet_test.expected
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
		echo "      http://mxnet.io/faq/env_var.html"
		echo " "
		echo "NEXT: Try the tutorials at: http://mxnet.io/tutorials"
		echo " "
		echo $LINE
		echo " "
		rm -f mxnet_test.log mxnet_test.expected
		return 0
	else
		echo " "
		echo "ERROR: Following files differ: mxnet_test.log mxnet_test.expected"
		echo "ERROR: MXNet test failed"
		echo "END: Test MXNet"
		echo " "
		echo ":-("
		exit 1
	fi
} # test_mxnet_python()

main() {
	print_intro_msg
	chk_mac_vers
	install_mac_pkg_manager
	install_dep_pip_for_mxnet
	chk_mxnet_installed
	download_mxnet
	compile_mxnet
	install_mxnet_python
	test_mxnet_python
} # main

main
