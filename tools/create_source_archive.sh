#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -e

MXNET_BRANCH=$1
MXNET_TAG=$2
CMD=$(basename $0)

if [[ -z "$MXNET_BRANCH" || -z "$MXNET_TAG" ]]; then
    echo "Usage: $CMD <branch> <tag>"
    echo "  where <branch> is the branch the tag was cut from."
    echo "        <tag> is the tag you want to build a release for."
    echo ""
    echo "  example: $CMD v1.8.x 1.8.0.rc3"
    exit -1
fi

TAR=tar
if [[ $(uname) == "Darwin" ]]; then
    TAR=gtar
fi

# make sure gnu tar is installed
which $TAR > /dev/null
if [[ $? -ne 0 ]]; then
    echo "It looks like you don't have GNU tar installed."
    echo ""
    echo "For OSX users, please install gnu-tar using the command 'brew install gnu-tar'"
    exit -1
fi

SRCDIR=apache-mxnet-src-$MXNET_TAG-incubating
TARBALL=$SRCDIR.tar.gz

# clone the repo and checkout the tag
echo "Cloning the MXNet repository..."
git clone -b $MXNET_BRANCH https://github.com/apache/incubator-mxnet.git $SRCDIR
pushd $SRCDIR
git submodule update --init --recursive
echo "Checking out tag $MXNET_TAG..."
git checkout $MXNET_TAG

echo "Removing unwanted artifacts..."
#### IMPORTANT ####
# Remove artifacts which do not comply with the Apache Licensing Policy
rm -rf R-package
rm -rf 3rdparty/mkldnn/doc

# Remove other artifacts we do not want contained in the source archive
rm -rf .DS_Store
rm -rf CODEOWNERS
find . -name ".git*" -print0 | xargs -0 rm -rf

# run Apache RAT license checker to verify all source files are compliant
echo "Running Apache RAT License Checker..."
ci/build.py -p ubuntu_rat /work/runtime_functions.sh nightly_test_rat_check

popd

echo "Creating tarball $TARBALL..."
$TAR -czf $TARBALL $SRCDIR

# sign the release tarball and create checksum file
gpg --armor --output $TARBALL.asc --detach-sig $TARBALL
shasum -a 512 $TARBALL > $TARBALL.sha512

