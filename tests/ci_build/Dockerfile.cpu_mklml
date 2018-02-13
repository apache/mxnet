FROM ubuntu:16.04

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

# Add MKLML library, compatiable with Ubuntu16.04
RUN wget --no-check-certificate -O /tmp/mklml.tgz https://github.com/01org/mkl-dnn/releases/download/v0.12/mklml_lnx_2018.0.1.20171227.tgz
RUN tar -zxvf /tmp/mklml.tgz && cp -rf mklml_*/* /usr/local/ && rm -rf mklml_*

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib
