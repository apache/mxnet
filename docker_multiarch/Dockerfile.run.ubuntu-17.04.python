# Before building this image you would need to build MXNet by executing:
# docker build -f Dockerfile.build.ubuntu-17.04 -t mxnet.build.ubuntu-17.04 .
# if you haven't done it before.

FROM mxnet.build.ubuntu-17.04

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update
RUN apt-get install -y python-pip
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /work/mxnet/python
RUN pip3 install -e .
RUN pip install -e .

