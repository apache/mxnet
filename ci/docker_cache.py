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
import tempfile
import build as build_util
import botocore
import subprocess
from botocore.handlers import disable_signing
from subprocess import call, check_call, CalledProcessError
from joblib import Parallel, delayed

S3_METADATA_IMAGE_ID_KEY = 'docker-image-id'

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

        # Compile and upload tarfile
        _compile_upload_cache_file(bucket_name='mxnet-ci-docker-cache-dev', docker_tag=docker_tag, image_id=image_id)
    except CalledProcessError as e:
        logging.error('Error during build of {}'.format(docker_tag))
        logging.exception(e)
        return


def _compile_upload_cache_file(bucket_name, docker_tag, image_id):
    session = _get_aws_session()
    s3_object = session.resource('s3').Object(bucket_name, docker_tag)

    remote_image_id = _get_remote_image_id(s3_object)
    if remote_image_id == image_id:
        logging.info('{} ({}) has not been updated - skipping'.format(docker_tag, image_id))
        return
    else:
        logging.debug('Cached image {} differs from local {}'.format(remote_image_id, image_id))

    # Compile layers into tarfile
    with tempfile.TemporaryDirectory() as temp_dir:
        tar_file_path = _format_docker_cache_filepath(output_dir=temp_dir, docker_tag=docker_tag)
        logging.debug('Writing layers of {} to {}'.format(docker_tag, tar_file_path))
        cmd = ['docker', 'save', docker_tag, '-o', tar_file_path]
        try:
            check_call(cmd)
        except CalledProcessError as e:
            logging.error('Error during save of {} at {}'.format(docker_tag, tar_file_path))
            return

        # Upload file
        logging.debug('Uploading {} to S3'.format(docker_tag))
        with open(tar_file_path, 'rb') as data:
            s3_object.upload_fileobj(Fileobj=data, ExtraArgs={"Metadata": {S3_METADATA_IMAGE_ID_KEY: image_id}})

def _get_remote_image_id(s3_object):
    """
    Get the image id of the docker cache which is represented by the S3 object
    :param s3_object: S3 object
    :return: Image id as string or None if object does not exist
    """
    try:
        if S3_METADATA_IMAGE_ID_KEY in s3_object.metadata:
            cached_image_id = s3_object.metadata[S3_METADATA_IMAGE_ID_KEY]
            return cached_image_id
        else:
            logging.debug('No cached image available')
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            logging.debug('{} does not exist in S3 yet'.format(s3_object))
        else:
            raise

    return None

def load_docker_cache(bucket_name, docker_tag):
    # Allow anonymous access
    s3_resource = boto3.resource('s3')
    s3_resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
    s3_object = s3_resource.Object(bucket_name, docker_tag)

    # Check if cache is still valid and exists
    remote_image_id = _get_remote_image_id(s3_object)
    if remote_image_id:
        if _docker_layer_exists(remote_image_id):
            logging.debug('Local docker cache already present')
            return
        else:
            logging.info('Local docker cache not present. TODO: Add instructions how to disable this')

        # Download using public S3 endpoint (without requiring credentials)
        with tempfile.TemporaryDirectory() as temp_dir:
            tar_file_path = os.path.join(temp_dir, 'layers.tar')
            s3_object.download_file(Filename=tar_file_path)

            # Load layers
            cmd = ['docker', 'load', '-i', tar_file_path]
            try:
                check_call(cmd)
            except CalledProcessError as e:
                logging.error('Error during load of docker cache for {} at {}'.format(docker_tag, tar_file_path))
                logging.exception(e)
                return
    else:
        logging.info('No cached remote image of {} present'.format(docker_tag))


def _docker_layer_exists(layer_id):
    """
    Check if the docker cache contains the layer with the passed id
    :param layer_id: layer id
    :return: True if exists, False otherwise
    """
    cmd = ['docker', 'images', '-q']
    image_ids_b = subprocess.check_output(cmd)
    image_ids_str = image_ids_b.decode('utf-8').strip()
    return layer_id in [id.strip() for id in image_ids_str.split('\n')]

def _get_aws_session() -> boto3.Session:  # pragma: no cover
    """
    Get the boto3 AWS session
    :return: Session object
    """
    global cached_aws_session
    if cached_aws_session:
        return cached_aws_session

    #session = boto3.Session(profile_name='mxnet-ci-dev', region_name='us-west-2')
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

    logging.getLogger().setLevel(logging.DEBUG)

    def script_name() -> str:
        return os.path.split(sys.argv[0])[1]

    logging.basicConfig(format='{}: %(asctime)-15s %(message)s'.format(script_name()))

    parser = argparse.ArgumentParser(description="Utility for preserving and loading Docker cache",epilog="")

    platforms = build_util.get_platforms()
    build_save_containers(platforms=platforms, output_dir='test')

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger('botocore').setLevel(logging.INFO)
    logging.getLogger('boto3').setLevel(logging.INFO)
    logging.getLogger('urllib3').setLevel(logging.INFO)
    load_docker_cache('mxnet-ci-docker-cache-dev', 'mxnet/build.jetson')
    load_docker_cache('mxnet-ci-docker-cache-dev', 'mxnet/build.ubuntu_cpu')
    sys.exit(main())
