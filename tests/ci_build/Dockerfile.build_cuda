FROM nvidia/cuda:8.0-cudnn5-devel
# cuda8.0 has to be used because this is the first ubuntu16.04 container
# which is required due to OpenBLAS being incompatible with ubuntu14.04
# the reason we used a gpu base container because we are going to test MKLDNN
# operator implementation against GPU implementation

COPY install/ubuntu_install_core.sh /install/
RUN /install/ubuntu_install_core.sh
COPY install/ubuntu_install_python.sh /install/
RUN /install/ubuntu_install_python.sh
COPY install/ubuntu_install_scala.sh /install/
RUN /install/ubuntu_install_scala.sh
COPY install/ubuntu_install_r.sh /install/
RUN /install/ubuntu_install_r.sh
COPY install/ubuntu_install_perl.sh /install/
RUN /install/ubuntu_install_perl.sh

# Allows to run tasks on a CPU without nvidia-docker and GPU 
COPY install/ubuntu_install_nvidia.sh /install/
RUN /install/ubuntu_install_nvidia.sh

# Add MKLML libraries
RUN wget --no-check-certificate -O /tmp/mklml.tgz https://github.com/01org/mkl-dnn/releases/download/v0.12/mklml_lnx_2018.0.1.20171227.tgz
RUN tar -zxvf /tmp/mklml.tgz && cp -rf mklml_*/* /usr/local/ && rm -rf mklml_*

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib
