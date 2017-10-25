FROM docker-registry.reminiz/nvidia_python3.6_opencv3.1_cuda8.0:master


ENV PATH=/opt/mxnet/files/commands:$PATH

ADD ./ /opt/mxnet

VOLUME /tmp/build_cache

CMD ["./opt/mxnet/files/commands/build"]

