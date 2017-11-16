#!/usr/bin/env bash
# install libraries for mxnet's scala package on ubuntu


apt-get install -y software-properties-common
add-apt-repository -y ppa:webupd8team/java
apt-get update
echo "oracle-java8-installer shared/accepted-oracle-license-v1-1 select true" | debconf-set-selections
apt-get install -y oracle-java8-installer
apt-get install -y oracle-java8-set-default

apt-get install -y maven 

wget http://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.deb
dpkg -i scala-2.11.8.deb
rm scala-2.11.8.deb
