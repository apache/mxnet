FROM nvidia/cuda:7.5-cudnn5-devel
# the reason we used a gpu base container because we are going to test MKLDNN
# operator implementation against GPU implementation

COPY install/ubuntu_install_core.sh /install/
RUN /install/ubuntu_install_core.sh
COPY install/ubuntu_install_python.sh /install/
RUN /install/ubuntu_install_python.sh
COPY install/ubuntu_install_scala.sh /install/
RUN /install/ubuntu_install_scala.sh

RUN wget --no-check-certificate -O /tmp/mklml.tgz https://github.com/01org/mkl-dnn/releases/download/v0.10/mklml_lnx_2018.0.20170908.tgz
RUN tar -zxvf /tmp/mklml.tgz && cp -rf mklml_*/* /usr/local/ && rm -rf mklml_*

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib
