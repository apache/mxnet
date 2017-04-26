# -*- mode: dockerfile -*-
# part of the dockerfile to install the python binding

COPY install/python.sh install/
RUN install/python.sh
RUN pip3 install jupyter
ENV PYTHONPATH=/mxnet/python
