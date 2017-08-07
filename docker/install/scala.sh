#!/usr/bin/env bash
# install libraries for mxnet's scala package on ubuntu

apt-get install -y maven default-jdk

wget http://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.deb
dpkg -i scala-2.11.8.deb
rm scala-2.11.8.deb
