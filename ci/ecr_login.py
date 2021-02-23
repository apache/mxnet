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
import logging
import os
import subprocess
import re
import sys
from typing import *

import build as build_util
from docker_login import login_dockerhub, logout_dockerhub
from util import retry



DOCKER_CACHE_NUM_RETRIES = 3
DOCKER_CACHE_TIMEOUT_MINS = 45
PARALLEL_BUILDS = 10
DOCKER_CACHE_RETRY_SECONDS = 5




ECR_LOGGED_IN = False
def _ecr_login(registry):
    """
    Use the AWS CLI to get credentials to login to ECR.
    """
    # extract region from registry
    global ECR_LOGGED_IN
    if ECR_LOGGED_IN:
        return
    regionMatch = re.match(r'.*?\.dkr\.ecr\.([a-z]+\-[a-z]+\-\d+)\.amazonaws\.com', registry)
    assert(regionMatch)
    region = regionMatch.group(1)
    logging.info("Logging into ECR region %s using aws-cli..", region)
    os.system("$(aws ecr get-login --region "+region+" --no-include-email)")
    ECR_LOGGED_IN = True


def main() -> int:
    # We need to be in the same directory than the script so the commands in the dockerfiles work as
    # expected. But the script can be invoked from a different path
    base = os.path.split(os.path.realpath(__file__))[0]
    os.chdir(base)

    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger('botocore').setLevel(logging.INFO)
    logging.getLogger('boto3').setLevel(logging.INFO)
    logging.getLogger('urllib3').setLevel(logging.INFO)
    logging.getLogger('s3transfer').setLevel(logging.INFO)
    
    def script_name() -> str:
        return os.path.split(sys.argv[0])[1]

    logging.basicConfig(format='{}: %(asctime)-15s %(message)s'.format(script_name()))
    
    parser = argparse.ArgumentParser(description="Utility for preserving and loading Docker cache", epilog="")
    parser.add_argument("--docker-registry",
                        help="Docker hub registry name",
                        type=str,
                        required=True)
    args = parser.parse_args()

    _ecr_login(args.docker_registry)


if __name__ == '__main__':
    sys.exit(main())

