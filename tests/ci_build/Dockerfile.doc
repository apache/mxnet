FROM ubuntu:14.04


COPY install/ubuntu_install_core.sh /install/
RUN /install/ubuntu_install_core.sh
COPY install/ubuntu_install_python.sh /install/
RUN /install/ubuntu_install_python.sh
COPY install/ubuntu_install_scala.sh /install/
RUN /install/ubuntu_install_scala.sh

RUN wget http://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.deb && \
    dpkg -i scala-2.11.8.deb && rm scala-2.11.8.deb

RUN apt-get install -y doxygen libatlas-base-dev graphviz pandoc
RUN pip install sphinx==1.3.5 CommonMark==0.5.4 breathe mock recommonmark pypandoc
