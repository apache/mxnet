#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

docker \
        run \
        -it \
        --cap-add \
        SYS_PTRACE \
        --rm \
        --shm-size=500m \
        -v \
        /home/ec2-user/work/incubator-mxnet:/work/mxnet \
        -v \
        /home/ec2-user/work/incubator-mxnet/build:/work/build \
        -v \
        /root/.ccache:/work/ccache \
        -u \
        0:0 \
        -e \
        CCACHE_MAXSIZE=500G \
        -e \
        CCACHE_TEMPDIR=/tmp/ccache \
        -e \
        CCACHE_DIR=/work/ccache \
        -e \
        CCACHE_LOGFILE=/tmp/ccache.log \
        -e \
        RELEASE_BUILD=false \
        mxnetci/build.ubuntu_cpu:latest \
        bash