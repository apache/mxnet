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
Utility to handle distributed docker cache
"""

import os
import logging
import argparse
import sys
import boto3
import build as build_util
from subprocess import call, check_call, CalledProcessError
from joblib import Parallel, delayed

cached_aws_session = None

def build_save_containers(platforms, output_dir):
    if len(platforms) == 0:
        return

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    Parallel(n_jobs=len(platforms), backend="threading")(
        delayed(_build_save_container)(platform, output_dir)
        for platform in platforms)

    output_files = []
    is_error = False
    # Ensure all containers have been built successfully and thus created the cache files
    for platform in platforms:
        docker_tag = build_util.get_docker_tag(platform)
        docker_cache_file = _format_docker_cache_filepath(output_dir=output_dir, docker_tag=docker_tag)

        # Check if cache file exists
        if not os.path.isfile(docker_cache_file):
            logging.error('Unable to find docker cache file at {} for {}'.format(docker_cache_file, docker_tag))
            is_error = True

        output_files.append(docker_cache_file)

    if is_error:
        sys.exit(1)

    return output_files


def _build_save_container(platform, output_dir):
    docker_tag = build_util.get_docker_tag(platform)
    docker_cache_file = _format_docker_cache_filepath(output_dir=output_dir, docker_tag=docker_tag)

    # Delete previous cache file
    try:
        os.remove(docker_cache_file)
    except OSError:
        pass

    # Start building
    logging.debug('Building {} as {}'.format(platform, docker_tag))
    try:
        image_id = build_util.build_docker(docker_binary='docker', platform=platform)
        logging.info('Built {} as {}'.format(docker_tag, image_id))
    except CalledProcessError as e:
        logging.error('Error during build of {}'.format(docker_tag))
        logging.exception(e)
        return

    # Compile layers into tarfile
    logging.debug('Writing layers of {} to {}'.format(docker_tag, docker_cache_file))
    cmd = ['docker', 'save', docker_tag, '-o', docker_cache_file]

    try:
        check_call(cmd)
    except CalledProcessError as e:
        logging.error('Error during save of {} at {}'.format(docker_tag, docker_cache_file))
        logging.exception(e)
        try:
            os.remove(docker_cache_file)
        except OSError:
            pass
        return

def upload_cache_file(bucket_name, cache_file):
    session = _get_aws_session()

    pass

def _get_aws_session() -> boto3.Session:  # pragma: no cover
    """
    Get the boto3 AWS session
    :return: Session object
    """
    global cached_aws_session
    if cached_aws_session:
        return cached_aws_session

    #session = boto3.Session(profile_name=SETTING_AWS_PROFILE_NAME, region_name=SETTING_AWS_REGION)
    session = boto3.Session()
    cached_aws_session = session
    return session

def _format_docker_cache_filepath(output_dir, docker_tag):
    return os.path.join(output_dir, docker_tag.replace('/', '_') + '.tar')

def main() -> int:
    # We need to be in the same directory than the script so the commands in the dockerfiles work as
    # expected. But the script can be invoked from a different path
    base = os.path.split(os.path.realpath(__file__))[0]
    os.chdir(base)

    logging.getLogger().setLevel(logging.INFO)

    def script_name() -> str:
        return os.path.split(sys.argv[0])[1]

    logging.basicConfig(format='{}: %(asctime)-15s %(message)s'.format(script_name()))

    parser = argparse.ArgumentParser(description="Utility for preserving and loading Docker cache",epilog="")

    platforms = build_util.get_platforms()
    build_save_containers(platforms=platforms, output_dir='test')

if __name__ == '__main__':
    sys.exit(main())
