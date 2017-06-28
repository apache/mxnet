#!/usr/bin/env bash
# install libraries for mxnet's r package on ubuntu

echo "deb http://cran.rstudio.com/bin/linux/ubuntu trusty/" >> /etc/apt/sources.list
gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
gpg -a --export E084DAB9 | apt-key add -

apt-get update
apt-get install -y r-base r-base-dev libxml2-dev libssl-dev

