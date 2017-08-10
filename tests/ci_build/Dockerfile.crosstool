FROM ubuntu
MAINTAINER Aran Khanna <arankhan@amazon.com>

# UPDATE BOX
RUN apt-get update && apt-get -y upgrade

# TOOLCHAIN DEPS
RUN apt-get install -y python python-setuptools python-pip python-dev unzip gfortran
RUN apt-get install -y git bison cvs flex gperf texinfo automake libtool help2man make libtool-bin libncurses5-dev g++ cmake wget gawk
RUN pip install numpy nose

# BUILD TOOLCHAIN
RUN git clone https://github.com/arank/crosstool-NG
RUN cd crosstool-NG && ./bootstrap && ./configure && make && make install

RUN useradd -ms /bin/bash aran
RUN cd && cp -R .profile .bashrc /home/aran
ADD . /home/aran/build

RUN chown -R aran:aran /home/aran

RUN cd /home/aran/build && su -m aran -c "export HOME=/home/aran;ct-ng arm-unknown-linux-gnueabi;ct-ng build"

