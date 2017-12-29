FROM nvidia/cuda:8.0-cudnn5-devel
# cuda8.0 has to be used because this is the first ubuntu16.04 container
# which is required due to OpenBLAS being incompatible with ubuntu14.04

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

# Fixes https://github.com/BVLC/caffe/issues/5658 See https://github.com/intel/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide
RUN echo "INCLUDE_DIRS += /usr/lib /usr/lib/x86_64-linux-gnu /usr/include/hdf5/serial/ " >> /caffe/Makefile.config
RUN echo "LIBRARY_DIRS += /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial " >> /caffe/Makefile.config

# Fixes https://github.com/BVLC/caffe/issues/4333 See https://github.com/intel/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide
# Note: This is only valid on Ubuntu16.04 - the version numbers are bound to the distribution
RUN ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial.so.10.0.2 /usr/lib/x86_64-linux-gnu/libhdf5.so
RUN ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so.10.0.2 /usr/lib/x86_64-linux-gnu/libhdf5_hl.so

RUN cd caffe; make all pycaffe -j$(nproc)

RUN cd caffe/python; for req in $(cat requirements.txt); do pip2 install $req; done

ENV PYTHONPATH=${PYTHONPATH}:/caffe/python
