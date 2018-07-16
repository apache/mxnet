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
# Tested on Ubuntu 16.04+ distro.
# Important Maintenance Instructions:
#  Align changes with CI in /ci/docker/install/ubuntu_r.sh
######################################################################
set -e
echo "This script assumes you have built MXNet already."

read -p "Would you like to continue? [Y/n]"
if [ x"$REPLY" = x"" -o x"$REPLY" = x"y" -o x"$REPLY" = x"Y" ]; then
  echo "Installing R dependencies. This can take few minutes..."

  sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
  sudo add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu xenial/'

  sudo apt-get -qq update
  sudo apt-get install -y --allow-unauthenticated \
      libcairo2-dev \
      libcurl4-openssl-dev \
      libssl-dev \
      libxml2-dev \
      libxt-dev \
      r-base \
      r-base-dev

  sudo Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
  cd ../../R-package
  sudo Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
  cd ..

  echo "Compiling R package. This can take few minutes..."
  sudo make rpkg

  echo "Done! MXNet for R installation is complete. Go ahead and explore MXNet with R :-)"
fi
