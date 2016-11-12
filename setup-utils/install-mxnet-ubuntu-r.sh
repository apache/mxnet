#!/usr/bin/env bash
######################################################################
# This script installs MXNet for R along with all required dependencies on a Ubuntu Machine.
# We recommend to install Microsoft RServer together with Intel MKL library for optimal performance
# More information can be found here: 
# https://blogs.technet.microsoft.com/machinelearning/2016/09/15/building-deep-neural-networks-in-the-cloud-with-azure-gpu-vms-mxnet-and-microsoft-r-server/
# Tested on Ubuntu 14.0+ distro.
# This script assumes that R is already installed in the system
######################################################################
set -e

MXNET_HOME=..

echo "Building MXNet core. This can take few minutes..."
cd $MXNET_HOME
make -j$(nproc)

echo "Installing R dependencies..."
sudo Rscript -e “install.packages(‘devtools’, repo = ‘https://cran.rstudio.com’)”
sudo Rscript -e “install.packages(c(‘Rcpp’, ‘DiagrammeR’, ‘data.table’, ‘jsonlite’, ‘magrittr’, ‘stringr’, ‘roxygen2’), repos = ‘https://cran.rstudio.com’)”
make rpkg
sudo R CMD INSTALL mxnet_0.7.tar.gz

echo "Done! MXNet for R installation is complete. Go ahead and explore MXNet with R :-)"