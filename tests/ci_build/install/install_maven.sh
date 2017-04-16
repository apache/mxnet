#!/usr/bin/env bash

set -e

wget http://mirrors.ocf.berkeley.edu/apache/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz
mv apache-maven-3.3.9-bin.tar.gz /tmp/
tar xfvz /tmp/apache-maven-3.3.9-bin.tar.gz

yum install -y java-1.8.0-openjdk-devel
