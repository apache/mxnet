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

import argparse
import json
import logging
import os
import subprocess
import sys

from util import retry, config_logging

DOCKERHUB_LOGIN_NUM_RETRIES = 5
DOCKERHUB_RETRY_SECONDS = 5


def _get_dockerhub_credentials(secret_name: str, secret_endpoint_url: str, secret_endpoint_region_name: str):
    import boto3
    import botocore

    session = boto3.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=secret_endpoint_region_name,
        endpoint_url=secret_endpoint_url
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
def login_dockerhub(secret_name: str, secret_endpoint_url: str, secret_endpoint_region_name: str):
    """
    Login to the Docker Hub account
    :return: None
    """
    dockerhub_credentials = _get_dockerhub_credentials(secret_name, secret_endpoint_url, secret_endpoint_region_name)

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
    if p.returncode == 0:
        logging.info('Successfully logged in to DockerHub')
        return

    raise RuntimeError("Failed to login to DockerHub")


def logout_dockerhub():
    """
    Log out of DockerHub to delete local credentials
    :return: None
    """
    logging.info('Logging out of DockerHub')
    subprocess.call(['docker', 'logout'])
    logging.info('Successfully logged out of DockerHub')


def main(command_line_arguments):
    config_logging()

    parser = argparse.ArgumentParser(
        description="Safe docker login utility to avoid leaking passwords",
        epilog=""
    )
    parser.add_argument("--secret-name",
                        help="Secret name",
                        type=str,
                        required=True)

    parser.add_argument("--secret-endpoint-url",
                        help="Endpoint Url",
                        type=str,
                        default=os.environ.get("DOCKERHUB_SECRET_ENDPOINT_URL", None))

    parser.add_argument("--secret-endpoint-region",
                        help="AWS Region",
                        type=str,
                        default=os.environ.get("DOCKERHUB_SECRET_ENDPOINT_REGION", None))

    args = parser.parse_args(args=command_line_arguments)

    if args.secret_endpoint_url is None:
        raise RuntimeError("Could not determine secret-endpoint-url, please specify with --secret-endpoint-url")

    if args.secret_endpoint_region is None:
        raise RuntimeError("Could not determine secret-endpoint-region, please specify with --secret-endpoint-region")

    try:
        login_dockerhub(args.secret_name, args.secret_endpoint_url, args.secret_endpoint_region)
    except Exception as err:
        logging.exception(err)
        exit(1)


if __name__ == '__main__':
    main(sys.argv[1:])
