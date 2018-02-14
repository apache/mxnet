#!/bin/bash

# set -ex
#
# All modification made by Intel Corporation: © 2016 Intel Corporation
#
# All contributions by the University of California:
# Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
# All rights reserved.
#
# All other contributions:
# Copyright (c) 2014, 2015, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
GetVersionName()
{
VERSION_LINE=0
if [ $1 ]; then
  VERSION_LINE=`grep __INTEL_MKL_BUILD_DATE $1/include/mkl_version.h 2>/dev/null | sed -e 's/.* //'`
fi
if [ -z $VERSION_LINE ]; then
  VERSION_LINE=0
fi
echo $VERSION_LINE  # Return Version Line
}

# MKL
HOME_MKL=$1
if [ ! -d "$HOME_MKL" ]; then
   mkdir $HOME_MKL
fi
MXNET_ROOT=`dirname $0`
USE_MKLML=0
# NOTE: if you update the following line, please also update the dockerfile at
# tests/ci_build/Dockerfile.mkl
VERSION_MATCH=20171227
PLATFORM=$(uname)
if [ $PLATFORM == "Darwin" ]; then
    INFIX=mac
elif [ $PLATFORM == "Linux" ]; then
    INFIX=lnx
fi
ARCHIVE_BASENAME=mklml_${INFIX}_2018.0.1.${VERSION_MATCH}.tgz
MKL_CONTENT_DIR=`echo $ARCHIVE_BASENAME | rev | cut -d "." -f 2- | rev`
MKLURL="https://github.com/01org/mkl-dnn/releases/download/v0.12/$ARCHIVE_BASENAME"
# there are diffrent MKL lib to be used for GCC and for ICC
reg='^[0-9]+$'
VERSION_LINE=`GetVersionName $MKLROOT`
#echo $VERSION_LINE
# Check if MKLROOT is set if positive then set one will be used..
if [ -z $MKLROOT ]; then
  # ..if MKLROOT is not set then check if we have MKL downloaded in proper version
    VERSION_LINE=`GetVersionName $HOME_MKL`
    #echo $VERSION_LINE
    if [ $VERSION_LINE -lt $VERSION_MATCH ] ; then
      #...If it is not then downloaded and unpacked
      if [ $PLATFORM == "Darwin" ]; then
        curl -L -o $MXNET_ROOT/$ARCHIVE_BASENAME $MKLURL
      elif [ $PLATFORM == "Linux" ]; then
        wget --quiet --no-check-certificate -P $MXNET_ROOT $MKLURL -O $MXNET_ROOT/$ARCHIVE_BASENAME
      fi
      tar -xzf $MXNET_ROOT/$ARCHIVE_BASENAME -C $MXNET_ROOT
      #echo $HOME_MKL
      yes | cp -rf $MXNET_ROOT/$MKL_CONTENT_DIR/* $HOME_MKL
      rm -rf $MXNET_ROOT/$MKL_CONTENT_DIR
    fi
  if [ $PLATFORM == "Darwin" ]; then
    MKLLIB=`find $HOME_MKL -name libmklml.dylib`
  elif [ $PLATFORM == "Linux" ]; then
    MKLLIB=`find $HOME_MKL -name libmklml_gnu.so`
  fi
  MKLROOT=`echo $MKLLIB | sed -e 's/lib.*$//'`
fi

# Check what MKL lib we have in MKLROOT
if [ -z `find $MKLROOT \( -name libmklml_gnu.so -o -name libmklml.dylib \) -print -quit` ]; then
  USE_MKLML=0
elif [ -z `find $MKLROOT -name libmkl_core.so -print -quit` ]; then
  USE_MKLML=1
fi

# return value to calling script (Makefile,cmake)
echo $MKLROOT $USE_MKLML
