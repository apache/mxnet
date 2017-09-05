#!/bin/bash
set -e
declare -a archs=("armv6" "armv7" "arm64" "ubuntu-17.04" "android.armv7" "ubuntu-16.04-cuda_8.0_cudnn5" "cmake.ubuntu-17.04")

read root < <(x=`pwd`; while [ "$x" != "/" ] ; do x=`dirname "$x"`; find "$x" -maxdepth 1 -name CONTRIBUTORS.md | xargs -I{} dirname {}; done)

if [ ! -d $root]; then
  echo "Root directory for mxnet not found"
  exit 1
else
  echo "Found root project $root"
fi

set -x
## For every arm architecture above build MXNet using a cross-compilation environment hosted in
## docker.  Then copy the build artifacts into an appropriate folder.

#if [ ! -d mxnet ]; then
#  rsync -a --delete --exclude=".git/" ../ mxnet
#fi
rsync -a --delete --exclude=".git/" --exclude "/docker/" $root/ mxnet

for i in "${archs[@]}"
do
  echo "***************************************"
  echo Building "$i"
  echo "***************************************"
  # Build mxnet within the appropriate container
  docker build -f Dockerfile.build.$i -t mxnet.build.$i  . || (echo "Build for $i failed" ; exit 1)
done

for i in "${archs[@]}"
do
  echo Copying Artifacts "$i"
  #mkdir -p $i
  # Copy the artifacts
  mkdir -p build/$i
  docker run -v `pwd`/build/$i:/$i mxnet.build.$i bash -c "cp /work/build/* /$i"
done
