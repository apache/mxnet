# Before building this image you would need to build MXNet by executing:
# docker build -f Dockerfile.build.ubuntu-17.04 -t mxnet.build.ubuntu-17.04 .
# if you haven't done it before.

FROM mxnet.build.ubuntu-17.04

# Scala package
WORKDIR /work
RUN wget --quiet http://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.deb
RUN dpkg -i scala-2.11.8.deb && rm scala-2.11.8.deb

WORKDIR /work/mxnet
RUN make scalapkg $BUILD_OPTS

WORKDIR /work/build/scala
RUN cp /work/mxnet/scala-package/core/target/*.jar .
RUN cp /work/mxnet/scala-package/assembly/linux-x86_64-cpu/target/*.jar .
