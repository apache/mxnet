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

# Install clang 3.9 (the same version as in XCode 8.*) and 5.0 (latest major release)
RUN wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-3.9 main" && \
    apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-5.0 main" && \
    apt-get update && \
    apt-get install -y clang-3.9 clang-5.0 && \
    clang-3.9 --version && \
    clang-5.0 --version
