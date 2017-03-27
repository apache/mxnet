#!/usr/bin/env bash
# install libraries for mxnet's r package on ubuntu

echo "deb http://cran.rstudio.com/bin/linux/ubuntu trusty/" >> /etc/apt/sources.list
gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
gpg -a --export E084DAB9 | apt-key add -

apt-get update
apt-get install -y r-base r-base-dev libxml2-dev libxt-dev libssl-dev

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f "./DESCRIPTION" ]; then
    cp ../../R-package/DESCRIPTION .
fi

Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
