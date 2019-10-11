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

import boto3
from botocore.exceptions import ClientError


def docker_login():
    """
    Login to the Docker Hub account
    :return: None
    """
    dockerhub_credentials = get_secret()

    logging.info('Logging in to DockerHub')
    p = subprocess.run(['docker', 'login', '--username', dockerhub_credentials['username'], '--password-stdin'],
                       stdout=subprocess.PIPE, input=str.encode(dockerhub_credentials['password']))
    logging.info(p.stdout)
    logging.info('Successfully logged in to DockerHub')


def get_secret():
    secret_name = os.environ['RELEASE_DOCKERHUB_SECRET_NAME']
    endpoint_url = os.environ['DOCKERHUB_SECRET_ENDPOINT_URL']
    region_name = os.environ['DOCKERHUB_SECRET_ENDPOINT_REGION']

    session = boto3.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name,
        endpoint_url=endpoint_url
    )

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            raise e
    else:
        return json.loads(get_secret_value_response['SecretString'])


if __name__ == '__main__':
    docker_login()
