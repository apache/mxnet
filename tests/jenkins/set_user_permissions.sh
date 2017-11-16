#!/bin/bash

# Exit script with error if any errors occur
set -e

if [[ ! $1 || ! $2 || ! $3 || ! $4 || ! $5 ]];
then
    echo "USAGE: " $(basename $"0") "USER_ID USER_NAME GROUP_ID GROUP_NAME SCRIPT"
    exit 1
fi

# Now set permissions to run the integration tests.
# These permission are required to do pip install of certain test dependencies.
# Example: pip install pyyaml for installing Keras in MXNet-Keras integration test.

USER_NAME=$2
GROUP_ID=$3
GROUP_NAME=$4
SCRIPT=$5

if [ -d “/usr/local/lib/“ ]; then
  chown -R ${USER_NAME}:${GROUP_NAME} /usr/local/lib/
fi
if [ -d “/usr/local/lib64/“ ]; then
  chown -R ${USER_NAME}:${GROUP_NAME} /usr/local/lib64/
fi
if [ -d “/usr/local/bin/“ ]; then
  chown -R ${USER_NAME}:${GROUP_NAME} /usr/local/bin/
fi
if [ -d “/opt/lib/“ ]; then
  chown -R ${USER_NAME}:${GROUP_NAME} /opt/lib/
fi

# Call run_as_user.sh script to set basic permission set.
bash -c tests/jenkins/run_as_user.sh $1 $2 $3 $4 $5
