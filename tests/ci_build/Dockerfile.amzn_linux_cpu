FROM amazonlinux
MAINTAINER Ly Nguyen <lynguyen@amazon.com>

COPY install/* /install/

RUN yum install -y git wget sudo

RUN chmod -R 755 install
RUN /install/install_opencv.sh
RUN /install/install_openblas.sh
RUN /install/install_python2.sh
RUN /install/install_python3.sh
RUN /install/install_testdeps.sh
RUN /install/install_julia.sh
RUN /install/install_maven.sh
RUN /install/install_library.sh
