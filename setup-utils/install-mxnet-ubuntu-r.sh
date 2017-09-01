#!/usr/bin/env bash

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

######################################################################
# This script installs MXNet for R along with all required dependencies on a Ubuntu Machine.
# Tested on Ubuntu 14.04+ distro.
######################################################################
set -e

MXNET_HOME="$HOME/mxnet/"
echo "MXNet root folder: $MXNET_HOME"

echo "Building MXNet core. This can take few minutes..."
cd "$MXNET_HOME"
make -j$(nproc)

echo "Installing R dependencies. This can take few minutes..."

# make sure we have essential tools installed
is_rscript_installed=$(which Rscript | wc -l)
if [ "$is_rscript_installed" = "0" ]; then
	read -p "Seems like Rscript is not installed. Install Rscript? [Y/n]"
	if [ x"$REPLY" = x"" -o x"$REPLY" = x"y" -o x"$REPLY" = x"Y" ]; then
		sudo add-apt-repository -y "deb http://cran.rstudio.com/bin/linux/ubuntu `lsb_release -cs`/"
		sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
		sudo apt-get -qq update
		sudo apt-get install -y r-base-core
	fi
fi

# libcurl4-openssl-dev and libssl-dev are needed for devtools.
sudo apt-get -y install libcurl4-openssl-dev libssl-dev

# Needed for R XML
sudo apt-get install libxml2-dev

# Needed for R Cairo
sudo apt-get install libxt-dev

sudo Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
cd R-package
sudo Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
cd ..

echo "Compiling R package. This can take few minutes..."
sudo make rpkg

echo "Installing R package..."
sudo R CMD INSTALL mxnet_current_r.tar.gz

echo "Done! MXNet for R installation is complete. Go ahead and explore MXNet with R :-)"
