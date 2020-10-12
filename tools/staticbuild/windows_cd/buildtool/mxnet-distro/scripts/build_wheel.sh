#!/usr/bin/env bash

cd mxnet-build
echo $(git rev-parse HEAD) >> python/mxnet/COMMIT_HASH
cd -

# Make wheel for testing
python setup.py bdist_wheel

# Now it's ready to test.
# After testing, Travis will build the wheel again
# The output will be in the 'dist' path.

set -eo pipefail
wheel_name=$(ls -t dist | head -n 1)
pip install -U --force-reinstall dist/$wheel_name
python sanity_test.py

# @szha: this is a workaround for travis-ci#6522
set +e
