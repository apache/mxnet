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
import sys

import boto3
from botocore.exceptions import ClientError


def post_wheel(path):
    """
    Posts mxnet wheel file to PyPI
    """
    logging.info('Posting {} to PyPI'.format(path))
    pypi_credentials = get_secret()

    cmd = 'python3 -m twine upload --username {} --password {} {}'.format(
        pypi_credentials['username'],
        pypi_credentials['password'],
        path)

    # The PyPI credentials for DEV has username set to 'skipPublish'
    # This way we do not attempt to publish the PyPI package
    # Just print a helpful message
    if pypi_credentials['username'] == 'skipPublish':
        print('In DEV account, skipping publish')
        print('Would have run: {}'.format(cmd))
        return 0
    else:
        # DO NOT PRINT CMD IN THIS BLOCK, includes password
        p = subprocess.run(cmd.split(' '),
                        stdout=subprocess.PIPE)
        logging.info(p.stdout)
        return p.returncode

def get_secret():
    secret_name = os.environ['CD_PYPI_SECRET_NAME']
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
    sys.exit(post_wheel(sys.argv[1]))
