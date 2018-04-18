FROM ubuntu:16.04
LABEL maintainer="markhama@amazon.com"

# Install dependencies
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    build-essential \
    ca-certificates \
    curl \
    doxygen \
    git \
    libatlas-base-dev \
    libjemalloc-dev \
    liblapack-dev \
    libopenblas-dev \
    libopencv-dev \
    pandoc \
    python-numpy \
    python-pip \
    software-properties-common \
    unzip \
    wget

# Setup Scala
RUN echo "deb https://dl.bintray.com/sbt/debian /" | tee -a /etc/apt/sources.list.d/sbt.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
RUN apt-get update && apt-get install -y \
    sbt \
    scala

RUN pip install --upgrade pip && pip install \
    beautifulsoup4 \
    breathe \
    CommonMark==0.5.4 \
    h5py \
    mock==1.0.1 \
    pypandoc \
    recommonmark==0.4.0 \
    sphinx==1.5.6


COPY *.sh /
COPY *.py /

RUN /build_site_tag.sh "1.1.0 1.0.0 0.12.1 0.12.0 0.11.0 master" 1.1.0 http://mxnet.incubator.apache.org/
