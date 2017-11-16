#!/usr/bin/env bash
# install libraries for mxnet's scala package on ubuntu

apt-get install -y software-properties-common
add-apt-repository -y ppa:webupd8team/java
apt-get update
echo "oracle-java8-installer shared/accepted-oracle-license-v1-1 select true" | debconf-set-selections
apt-get install -y oracle-java8-installer
apt-get install -y oracle-java8-set-default
apt-get update && apt-get install -y maven
