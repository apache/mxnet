#!/bin/bash

# Exit script with error if any errors occur
set -e

if [[ ! $1 || ! $2 || ! $3 || ! $4 || ! $5 ]];
then
    echo "USAGE: " $(basename $"0") "USER_ID USER_NAME GROUP_ID GROUP_NAME SCRIPT"
    exit 1
fi

USER_ID=$1
USER_NAME=$2
GROUP_ID=$3
GROUP_NAME=$4
SCRIPT=$5

HOME_DIR=/home/${USER_NAME}

groupadd -f -g ${GROUP_ID} ${GROUP_NAME}
useradd -m -u ${USER_ID} -g ${GROUP_NAME} ${USER_NAME}
chown -R ${USER_NAME}:${GROUP_NAME} ${HOME_DIR}
chown -R ${USER_NAME}:${GROUP_NAME} /usr/local/lib/
echo "%${GROUP_NAME}  ALL=(ALL)       NOPASSWD: ALL" >> /etc/sudoers
su -m ${USER_NAME} -c "export HOME=${HOME_DIR}; ${SCRIPT}"
