#!/usr/bin/env bash
######################################################################
# This script installs MXNet for R along with all required dependencies on a Ubuntu Machine.
# We recommend to install Microsoft RServer together with Intel MKL library for optimal performance
# More information can be found here:
# https://blogs.technet.microsoft.com/machinelearning/2016/09/15/building-deep-neural-networks-in-the-cloud-with-azure-gpu-vms-mxnet-and-microsoft-r-server/
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
