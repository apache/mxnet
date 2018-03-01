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
