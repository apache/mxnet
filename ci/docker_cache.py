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

import os
import logging
import argparse
import sys
import boto3
import tempfile
import pprint
import threading
import build as build_util
import botocore
import subprocess
from botocore.handlers import disable_signing
from subprocess import call, check_call, CalledProcessError
from joblib import Parallel, delayed

S3_METADATA_IMAGE_ID_KEY = 'docker-image-id'
LOG_PROGRESS_PERCENTAGE_THRESHOLD = 10

cached_aws_session = None


class ProgressPercentage(object):
    def __init__(self, object_name, size):
        self._object_name = object_name
        self._size = size
        self._seen_so_far = 0
        self._last_percentage = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount) -> None:
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = int((self._seen_so_far / self._size) * 100)
            if (percentage - self._last_percentage) >= LOG_PROGRESS_PERCENTAGE_THRESHOLD:
                self._last_percentage = percentage
                logging.info('{}% of {}'.format(percentage, self._object_name))


def build_save_containers(platforms, bucket) -> int:
    """
    Entry point to build and upload all built dockerimages in parallel
    :param platforms: List of platforms
    :param bucket: S3 bucket name
    :return: 1 if error occurred, 0 otherwise
    """
    if len(platforms) == 0:
        return 0

    platform_results = Parallel(n_jobs=len(platforms), backend="multiprocessing")(
        delayed(_build_save_container)(platform, bucket)
        for platform in platforms)

    is_error = False
    for platform_result in platform_results:
        if platform_result is not None:
            logging.error('Failed to generate {}'.format(platform_result))
            is_error = True

    return 1 if is_error else 0


def _build_save_container(platform, bucket) -> str:
    """
    Build image for passed platform and upload the cache to the specified S3 bucket
    :param platform: Platform
    :param bucket: Target s3 bucket
    :return: Platform if failed, None otherwise
    """
    docker_tag = build_util.get_docker_tag(platform)

    # Preload cache
    # TODO: Allow to disable this in order to allow clean rebuilds
    load_docker_cache(bucket_name=bucket, docker_tag=docker_tag)

    # Start building
    logging.debug('Building {} as {}'.format(platform, docker_tag))
    try:
        image_id = build_util.build_docker(docker_binary='docker', platform=platform)
        logging.info('Built {} as {}'.format(docker_tag, image_id))

        # Compile and upload tarfile
        _compile_upload_cache_file(bucket_name=bucket, docker_tag=docker_tag, image_id=image_id)
        return None
    except Exception:
        logging.exception('Unexpected exception during build of {}'.format(docker_tag))
        return platform
        # Error handling is done by returning the errorous platform name. This is necessary due to
        # Parallel being unable to handle exceptions


def _compile_upload_cache_file(bucket_name, docker_tag, image_id) -> None:
    """
    Upload the passed image by id, tag it with docker tag and upload to S3 bucket
    :param bucket_name: S3 bucket name
    :param docker_tag: Docker tag
    :param image_id: Image id
    :return: None
    """
    session = _get_aws_session()
    s3_object = session.resource('s3').Object(bucket_name, docker_tag)

    remote_image_id = _get_remote_image_id(s3_object)
    if remote_image_id == image_id:
        logging.info('{} ({}) for {} has not been updated - skipping'.format(docker_tag, image_id, docker_tag))
        return
    else:
        logging.debug('Cached image {} differs from local {} for {}'.format(remote_image_id, image_id, docker_tag))

    # Compile layers into tarfile
    with tempfile.TemporaryDirectory() as temp_dir:
        tar_file_path = _format_docker_cache_filepath(output_dir=temp_dir, docker_tag=docker_tag)
        logging.debug('Writing layers of {} to {}'.format(docker_tag, tar_file_path))
        history_cmd = ['docker', 'history', '-q', docker_tag]

        image_ids_b = subprocess.check_output(history_cmd)
        image_ids_str = image_ids_b.decode('utf-8').strip()
        layer_ids = [id.strip() for id in image_ids_str.split('\n') if id != '<missing>']

        # docker_tag is important to preserve the image name. Otherwise, the --cache-from feature will not be able to
        # reference the loaded cache later on. The other layer ids are added to ensure all intermediary layers
        # are preserved to allow resuming the cache at any point
        cmd = ['docker', 'save', '-o', tar_file_path, docker_tag]
        cmd.extend(layer_ids)
        try:
            check_call(cmd)
        except CalledProcessError as e:
            logging.error('Error during save of {} at {}. Command: {}'.
                          format(docker_tag, tar_file_path, pprint.pprint(cmd)))
            return

        # Upload file
        logging.info('Uploading {} to S3'.format(docker_tag))
        with open(tar_file_path, 'rb') as data:
            s3_object.upload_fileobj(
                Fileobj=data,
                Callback=ProgressPercentage(object_name=docker_tag, size=os.path.getsize(tar_file_path)),
                ExtraArgs={"Metadata": {S3_METADATA_IMAGE_ID_KEY: image_id}})
            logging.info('Uploaded {} to S3'.format(docker_tag))


def _get_remote_image_id(s3_object) -> str:
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
            logging.debug('No cached image available for {}'.format(s3_object.key))
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            logging.debug('{} does not exist in S3 yet'.format(s3_object.key))
        else:
            raise

    return None


def load_docker_cache(bucket_name, docker_tag) -> None:
    """
    Load the precompiled docker cache from the passed S3 bucket
    :param bucket_name: S3 bucket name
    :param docker_tag: Docker tag to load
    :return: None
    """
    # Allow anonymous access
    s3_resource = boto3.resource('s3')
    s3_resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
    s3_object = s3_resource.Object(bucket_name, docker_tag)

    # Check if cache is still valid and exists
    remote_image_id = _get_remote_image_id(s3_object)
    if remote_image_id:
        if _docker_layer_exists(remote_image_id):
            logging.info('Local docker cache already present for {}'.format(docker_tag))
            return
        else:
            logging.info('Local docker cache not present for {}'.format(docker_tag))

        # Download using public S3 endpoint (without requiring credentials)
        with tempfile.TemporaryDirectory() as temp_dir:
            tar_file_path = os.path.join(temp_dir, 'layers.tar')
            s3_object.download_file(
                Filename=tar_file_path,
                Callback=ProgressPercentage(object_name=docker_tag, size=s3_object.content_length))

            # Load layers
            cmd = ['docker', 'load', '-i', tar_file_path]
            try:
                check_call(cmd)
                logging.info('Docker cache for {} loaded successfully'.format(docker_tag))
            except CalledProcessError as e:
                logging.error('Error during load of docker cache for {} at {}'.format(docker_tag, tar_file_path))
                logging.exception(e)
                return
    else:
        logging.info('No cached remote image of {} present'.format(docker_tag))


def _docker_layer_exists(layer_id) -> bool:
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

    session = boto3.Session()  # Uses IAM user credentials
    cached_aws_session = session
    return session


def _format_docker_cache_filepath(output_dir, docker_tag) -> str:
    return os.path.join(output_dir, docker_tag.replace('/', '_') + '.tar')


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

    parser = argparse.ArgumentParser(description="Utility for preserving and loading Docker cache",epilog="")
    parser.add_argument("--docker-cache-bucket",
                        help="S3 docker cache bucket, e.g. mxnet-ci-docker-cache",
                        type=str,
                        required=True)

    args = parser.parse_args()

    platforms = build_util.get_platforms()
    _get_aws_session()  # Init AWS credentials
    return build_save_containers(platforms=platforms, bucket=args.docker_cache_bucket)


if __name__ == '__main__':
    sys.exit(main())
