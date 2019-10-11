#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import json
import logging
import os
import subprocess
from util import retry

DOCKERHUB_LOGIN_NUM_RETRIES = 5
DOCKERHUB_RETRY_SECONDS = 5


def _get_dockerhub_credentials():  # pragma: no cover
    import boto3
    import botocore
    secret_name = os.environ['DOCKERHUB_SECRET_NAME']
    endpoint_url = os.environ['DOCKERHUB_SECRET_ENDPOINT_URL']
    region_name = os.environ['DOCKERHUB_SECRET_ENDPOINT_REGION']

    session = boto3.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name,
        endpoint_url=endpoint_url
    )
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except botocore.exceptions.ClientError as client_error:
        if client_error.response['Error']['Code'] == 'ResourceNotFoundException':
            logging.exception("The requested secret %s was not found", secret_name)
        elif client_error.response['Error']['Code'] == 'InvalidRequestException':
            logging.exception("The request was invalid due to:")
        elif client_error.response['Error']['Code'] == 'InvalidParameterException':
            logging.exception("The request had invalid params:")
        raise
    else:
        secret = get_secret_value_response['SecretString']
        secret_dict = json.loads(secret)
        return secret_dict


@retry(target_exception=subprocess.CalledProcessError, tries=DOCKERHUB_LOGIN_NUM_RETRIES,
       delay_s=DOCKERHUB_RETRY_SECONDS)
def login_dockerhub():
    """
    Login to the Docker Hub account
    :return: None
    """
    dockerhub_credentials = _get_dockerhub_credentials()

    logging.info('Logging in to DockerHub')
    # We use password-stdin instead of --password to avoid leaking passwords in case of an error.
    # This method will produce the following output:
    # > WARNING! Your password will be stored unencrypted in /home/jenkins_slave/.docker/config.json.
    # > Configure a credential helper to remove this warning. See
    # > https://docs.docker.com/engine/reference/commandline/login/#credentials-store
    # Since we consider the restricted slaves a secure environment, that's fine. Also, using this will require
    # third party applications which would need a review first as well.
    p = subprocess.run(['docker', 'login', '--username', dockerhub_credentials['username'], '--password-stdin'],
                       stdout=subprocess.PIPE, input=str.encode(dockerhub_credentials['password']))
    logging.info(p.stdout)
    logging.info('Successfully logged in to DockerHub')


def logout_dockerhub():
    """
    Log out of DockerHub to delete local credentials
    :return: None
    """
    logging.info('Logging out of DockerHub')
    subprocess.call(['docker', 'logout'])
    logging.info('Successfully logged out of DockerHub')


if __name__ == '__main__':
    login_dockerhub()
