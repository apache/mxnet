FROM nvidia/cuda:7.5-cudnn5-devel

COPY install/ubuntu_install_core.sh /install/
RUN /install/ubuntu_install_core.sh

COPY install/ubuntu_install_python.sh /install/
RUN /install/ubuntu_install_python.sh

RUN apt-get install -y libprotobuf-dev libleveldb-dev \
    libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler \
    libatlas-base-dev python-dev libgflags-dev libgoogle-glog-dev liblmdb-dev \
    python-numpy python-opencv

RUN apt-get install -y --no-install-recommends libboost-all-dev

RUN cd /; git clone http://github.com/BVLC/caffe.git; cd caffe; \
    cp Makefile.config.example Makefile.config

RUN echo "CPU_ONLY := 1" >> /caffe/Makefile.config

RUN cd caffe; make all pycaffe -j$(nproc)

RUN cd caffe/python; for req in $(cat requirements.txt); do pip2 install $req; done

ENV PYTHONPATH=${PYTHONPATH}:/caffe/python
