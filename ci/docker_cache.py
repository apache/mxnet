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

"""
Utility to handle distributed docker cache. This is done by keeping the entire image chain of a docker container
on an S3 bucket. This utility allows cache creation and download. After execution, the cache will be in an identical
state as if the container would have been built locally already.
"""

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


def build_save_containers(platforms, registry, load_cache) -> int:
    """
    Entry point to build and upload all built dockerimages in parallel
    :param platforms: List of platforms
    :param registry: Docker registry name
    :param load_cache: Load cache before building
    :return: 1 if error occurred, 0 otherwise
    """
    from joblib import Parallel, delayed
    if len(platforms) == 0:
        return 0

    platform_results = Parallel(n_jobs=PARALLEL_BUILDS, backend="multiprocessing")(
        delayed(_build_save_container)(platform, registry, load_cache)
        for platform in platforms)

    is_error = False
    for platform_result in platform_results:
        if platform_result is not None:
            logging.error('Failed to generate %s', platform_result)
            is_error = True

    return 1 if is_error else 0


def _build_save_container(platform, registry, load_cache) -> Optional[str]:
    """
    Build image for passed platform and upload the cache to the specified S3 bucket
    :param platform: Platform
    :param registry: Docker registry name
    :param load_cache: Load cache before building
    :return: Platform if failed, None otherwise
    """
    # Case 1: docker-compose
    if platform in build_util.DOCKER_COMPOSE_WHITELIST:
        build_util.build_docker(platform=platform, registry=registry, num_retries=10, no_cache=False)
        push_cmd = ['docker-compose', 'push', platform]
        subprocess.check_call(push_cmd)
        return None

    # Case 2: Deprecated way, will be removed
    docker_tag = build_util.get_docker_tag(platform=platform, registry=registry)

    # Preload cache
    if load_cache:
        load_docker_cache(registry=registry, docker_tag=docker_tag)

    # Start building
    logging.debug('Building %s as %s', platform, docker_tag)
    try:
        # Increase the number of retries for building the cache.
        image_id = build_util.build_docker(platform=platform, registry=registry, num_retries=10, no_cache=False)
        logging.info('Built %s as %s', docker_tag, image_id)

        # Push cache to registry
        _upload_image(registry=registry, docker_tag=docker_tag, image_id=image_id)
        return None
    except Exception:
        logging.exception('Unexpected exception during build of %s', docker_tag)
        return platform
        # Error handling is done by returning the errorous platform name. This is necessary due to
        # Parallel being unable to handle exceptions

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

def _upload_image(registry, docker_tag, image_id) -> None:
    """
    Upload the passed image by id, tag it with docker tag and upload to docker registry.
    :param registry: Docker registry name
    :param docker_tag: Docker tag
    :param image_id: Image id
    :return: None
    """
    if "dkr.ecr" in registry:
        _ecr_login(registry)

    # We don't have to retag the image since it is already in the right format
    logging.info('Uploading %s (%s) to %s', docker_tag, image_id, registry)
    push_cmd = ['docker', 'push', docker_tag]
    subprocess.check_call(push_cmd)


@retry(target_exception=subprocess.TimeoutExpired, tries=DOCKER_CACHE_NUM_RETRIES,
       delay_s=DOCKER_CACHE_RETRY_SECONDS)
def load_docker_cache(registry, docker_tag) -> None:
    """
    Load the precompiled docker cache from the registry
    :param registry: Docker registry name
    :param docker_tag: Docker tag to load
    :return: None
    """
    # We don't have to retag the image since it's already in the right format
    if not registry:
        return
    assert docker_tag

    if "dkr.ecr" in registry:
        _ecr_login(registry)

    logging.info('Loading Docker cache for %s from %s', docker_tag, registry)
    pull_cmd = ['docker', 'pull', docker_tag]

    # Don't throw an error if the image does not exist
    subprocess.run(pull_cmd, timeout=DOCKER_CACHE_TIMEOUT_MINS*60)
    logging.info('Successfully pulled docker cache')


def delete_local_docker_cache(docker_tag):
    """
    Delete the local docker cache for the entire docker image chain
    :param docker_tag: Docker tag
    :return: None
    """
    history_cmd = ['docker', 'history', '-q', docker_tag]

    try:
        image_ids_b = subprocess.check_output(history_cmd)
        image_ids_str = image_ids_b.decode('utf-8').strip()
        layer_ids = [id.strip() for id in image_ids_str.split('\n') if id != '<missing>']

        delete_cmd = ['docker', 'image', 'rm', '--force']
        delete_cmd.extend(layer_ids)
        subprocess.check_call(delete_cmd)
    except subprocess.CalledProcessError as error:
        # Could be caused by the image not being present
        logging.debug('Error during local cache deletion %s', error)


def main() -> int:
    """
    Utility to create and publish the Docker cache to Docker Hub
    :return:
    """
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

    platforms = build_util.get_platforms()

    if "dkr.ecr" in args.docker_registry:
        _ecr_login(args.docker_registry)
        return build_save_containers(platforms=platforms, registry=args.docker_registry, load_cache=True)
    else:
        secret_name = os.environ['DOCKERHUB_SECRET_NAME']
        endpoint_url = os.environ['DOCKERHUB_SECRET_ENDPOINT_URL']
        region_name = os.environ['DOCKERHUB_SECRET_ENDPOINT_REGION']

        try:
            login_dockerhub(secret_name, endpoint_url, region_name)
            return build_save_containers(platforms=platforms, registry=args.docker_registry, load_cache=True)
        finally:
            logout_dockerhub()


if __name__ == '__main__':
    sys.exit(main())
