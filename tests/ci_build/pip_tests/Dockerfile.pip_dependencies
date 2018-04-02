# -*- mode: dockerfile -*-
# part of the dockerfile to test pip installations

# add repo to install different Python versions
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update
RUN apt-get install -y python python2.7 python3.4 python3.5 python3.6

# install other dependencies
RUN apt-get install -y wget git unzip gcc
RUN apt-get install -y libgfortran3

# install virtualenv
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && pip install virtualenv && rm -rf get-pip.py
