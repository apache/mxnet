FROM nvidia/cuda:7.5-cudnn5-devel

COPY install/ubuntu_*.sh /install/

RUN /install/ubuntu_install_core.sh
RUN /install/ubuntu_install_python.sh
RUN /install/ubuntu_install_scala.sh
