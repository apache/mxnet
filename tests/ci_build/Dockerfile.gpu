FROM nvidia/cuda:8.0-cudnn5-devel
# cuda8.0 has to be used because this is the first ubuntu16.04 container
# which is required due to OpenBLAS being incompatible with ubuntu14.04

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
