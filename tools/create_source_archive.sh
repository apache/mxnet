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

MXNET_TAG=$1
CMD=$(basename $0)

if [[ -z "$MXNET_TAG" ]]; then
    echo "Usage: $CMD <tag>"
    echo "  where <tag> is the git tag you want to build a release for."
    echo ""
    echo "  example: $CMD 1.8.0.rc3"
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
git clone -b $MXNET_TAG --depth 1 --recurse-submodules \
	--shallow-submodules https://github.com/apache/incubator-mxnet.git \
	$SRCDIR
pushd $SRCDIR

echo "Removing unwanted artifacts..."
#### IMPORTANT ####
# Remove artifacts which do not comply with the Apache Licensing Policy
for d in $(cat tools/source-exclude-artifacts.txt | grep -v "^#"); do
	if [[ -e $d ]]; then
		echo "Removing $d from source archive..."
		rm -rf $d
	fi
done

# Remove other artifacts we do not want contained in the source archive
rm -rf .DS_Store
rm -rf CODEOWNERS
rm -rf .github

# run Apache RAT license checker to verify all source files are compliant
echo "Running Apache RAT License Checker..."
ci/build.py -p ubuntu_rat /work/runtime_functions.sh nightly_test_rat_check

popd

echo "Creating tarball $TARBALL..."
$TAR --exclude-vcs -czf $TARBALL $SRCDIR

# sign the release tarball and create checksum file
gpg --armor --output $TARBALL.asc --detach-sig $TARBALL
shasum -a 512 $TARBALL > $TARBALL.sha512

