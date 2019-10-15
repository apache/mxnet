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
Tool for uploading artifacts to the artifact repository
"""

__author__ = 'Per Goncalves da Silva'
__version__ = '0.1'

import argparse
import ctypes
import glob
import logging
import os
import re
import sys
from itertools import chain
from subprocess import CalledProcessError, check_output
from typing import Dict, List, Optional

import boto3
import botocore.exceptions
import yaml

s3 = boto3.client('s3')
logger = logging.getLogger(__name__)


def config_logging():
    """
    Configures default logging settings
    """
    logging.root.setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(levelname)s: %(message)s')


def s3_upload(bucket: str, s3_key_prefix: str, paths: List[str]):
    """
    Uploads a list of files to an S3 bucket with a particular S3 key prefix
    :param bucket: The name of the S3 bucket
    :param s3_key_prefix: The key prefix to apply to each of the files
    :param paths: A list of paths to files
    """
    for path in paths:
        s3_key = "{}/{}".format(s3_key_prefix, os.path.basename(path))
        logger.info('Uploading {}'.format(path))
        logger.debug("Uploading {} to s3://{}/{}".format(path, bucket, s3_key))
        with open(path, 'rb') as data:
            s3.upload_fileobj(Fileobj=data, Key=s3_key, Bucket=bucket)


def write_libmxnet_meta(args: argparse.Namespace, destination: str):
    """
    Writes a file called libmxnet.meta in the 'destination' folder that contains
    the libmxnet library information (commit id, type, etc.).
    :param args: A Namespace object containing the library
    :param destination: The folder in which to place the libmxnet.meta
    """
    with open(os.path.join(destination, 'libmxnet.meta'), 'w') as fp:
        fp.write(yaml.dump({
            "variant": args.variant,
            "os": args.os,
            "commit_id": args.git_sha,
            "dependency_linking": args.libtype,
        }))


def try_s3_download(bucket, s3_key_prefix, destination) -> bool:
    """
    Downloads a list of files to an S3 bucket with a particular S3 key prefix to 'destination'
    :param bucket: The name of the S3 bucket
    :param s3_key_prefix: The key prefix to apply to each of the files
    :param destination the path to which to download the files
    :return False if not artifacts were found, True otherwise
    """
    response = s3.list_objects_v2(Bucket=bucket, Prefix=s3_key_prefix)
    if not response:
        raise RuntimeError('Error listing S3 objects')

    if response.get('KeyCount') is None:
        logger.debug('Invalid S3 list objects response format')
        logger.debug(response)
        raise RuntimeError('Invalid response format.')

    key_count = response.get('KeyCount')
    if key_count == 0:
        logger.debug('No artifacts found')
        return False

    if not response.get('Contents'):
        logger.debug('Invalid S3 list objects response format')
        logger.debug(response)
        raise RuntimeError('Invalid response format.')

    for obj in response.get('Contents'):
        key = obj['Key']

        # extract file path with any subdirectories and remove the leading file separator
        output_path = os.path.join(destination, key[len(s3_key_prefix):].lstrip(os.sep))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info('Downloading {}'.format(output_path))
        logger.debug("Downloading s3://{}/{} to {}".format(bucket, key, output_path))
        with open(output_path, 'wb') as fp:
            s3.download_fileobj(Fileobj=fp, Key=key, Bucket=bucket)

    return True


def get_commit_id_from_cmd() -> Optional[str]:
    """
    Returns the output of 'git rev-parse HEAD'
    :return: A commit id, or None if the command fails
    """
    try:
        logger.debug('Executing "git rev-parse HEAD"')
        commit_id = check_output("git rev-parse HEAD".split(" ")).decode('UTF-8').strip()
        logger.debug('Found commit id: {}'.format(commit_id))
        return commit_id
    except CalledProcessError as e:
        logger.debug('Error getting commit id:')
        logger.debug(format(e))
        return None


def probe_commit_id() -> str:
    """
    Probes the system in an attempt to ascertain the mxnet commit id
    :return: The commit id, or None if not found
    """
    logger.debug('Probing for commit id')
    commit_id = os.environ.get('MXNET_SHA')
    if not commit_id:
        logger.debug('MXNET_SHA environment variable not set. Trying GIT_COMMIT')
        commit_id = os.environ.get('GIT_COMMIT')
    if not commit_id:
        logger.debug('GIT_COMMIT environment variable not set. Trying git command')
        commit_id = get_commit_id_from_cmd()
    if not commit_id:
        logger.debug('Could not determine git commit id')
    else:
        logger.debug('Commit id is: {}'.format(commit_id))
    return commit_id


def get_linux_os_release_properties() -> Optional[Dict[str, str]]:
    """
    Makes a dictionary out of /etc/os-release
    :return: A dictionary of os release properties
    """
    logger.debug('Extracting operating system properties from /etc/os-release')
    if not os.path.isfile('/etc/os-release'):
        logger.debug('Error: /etc/os-release not found')
        return None

    try:
        with open('/etc/os-release', 'r') as fp:
            # removes empty spaces and quotation marks from line
            property_tuple_list = [line.strip().replace('"', '').split('=') for line in fp if line.strip()]
            return {key: value for (key, value) in property_tuple_list}
    except Exception as e:
        logger.debug('Error parsing /etc/os-release')
        logger.debug(e)
        return None


def get_linux_distribution_and_version() -> Optional[str]:
    """
    Returns the linux distribution and version by taking
    the values of ID and VERSION_ID from /etc/os-release and
    concatenating them. Eg. centos7, ubuntu16.04, etc.
    :return: The linux distribution and version string, or None if not found.
    """
    logger.debug('Getting linux distribution and version')
    os_properties = get_linux_os_release_properties()
    if os_properties:
        logger.debug('os properties: {}'.format(os_properties))
        distribution = os_properties['ID']
        version = os_properties['VERSION_ID']
        return "{}{}".format(distribution, version)

    logger.debug('Error getting linux distribution and version. Could not determine os properties.')
    return None


def probe_operating_system() -> str:
    """
    Probes the system to determine the operating system.
    :return: The name of the operating system, e.g. win32, darwin, ubuntu16.04, centos7, etc.
    """
    logger.debug('Determining operating system')
    operating_system = sys.platform
    logger.debug('Found platform: {}'.format(operating_system))
    if operating_system.startswith('linux'):
        operating_system = get_linux_distribution_and_version()

    logger.debug('Operating system is {}'.format(operating_system))
    return operating_system


def get_libmxnet_features(libmxnet_path: str) -> Optional[Dict[str, bool]]:
    """
    Returns a string -> boolean dictionary mapping feature name
    to whether it is enabled or not
    :param libmxnet_path: path to the libmxnet library
    :return: dictionary of features to whether they are enabled
    """
    logger.debug('Getting feature dictionary from {}'.format(libmxnet_path))

    class Feature(ctypes.Structure):
        _fields_ = [("_name", ctypes.c_char_p), ("enabled", ctypes.c_bool)]

        @property
        def name(self):
            return self._name.decode()

    # we are not using the mxnet python bindings here because we cannot assume
    # they are present and in the python path, or that they would point to the
    # specified libmxnet.so. Therefore, we load the libmxnet.so library independently
    # to extract its features
    try:
        libmxnet = ctypes.CDLL(libmxnet_path, ctypes.RTLD_LOCAL)
    except Exception as e:
        logger.error('Error loading {}. '
                     'Please check check path to libmxnet library is correct.'.format(libmxnet_path))
        logger.error(e)
        return None

    libmxnet.MXGetLastError.restype = ctypes.c_char_p
    feature_array = ctypes.POINTER(Feature)()
    feature_array_size = ctypes.c_size_t()
    if libmxnet.MXLibInfoFeatures(ctypes.byref(feature_array), ctypes.byref(feature_array_size)) != 0:
        logger.error('Could not determine features from {}. '
                     'Please specify the variant manually using the "--variant" argument.'.format(libmxnet_path))
        return None
    features = {feature_array[i].name: feature_array[i].enabled for i in range(feature_array_size.value)}
    logger.debug('Found features: {}'.format(features))
    return features


def get_cuda_version() -> Optional[str]:
    """
    Returns the major and minor cuda version without the '.'
    eg. 10.0 => 100, 9.2 => 92, etc.
    :return: CUDA version
    """
    logger.debug('Determining cuda version')

    try:
        logger.debug('Executing "nvcc -V"')
        nvcc_version = check_output("nvcc -V".split(" ")).decode('UTF-8').strip()
    except CalledProcessError as e:
        logger.error('Error getting nvcc version')
        logger.error(e)
        return None

    logger.debug('Extracting cuda version from {}'.format(nvcc_version))
    # eg. "Cuda compilation tools, release 10.0, V10.0.130"
    match = re.search(r' ([0-9]+.[0-9]+)', nvcc_version)
    if match:
        cuda_version = match.group(1).replace('.', '')
        logger.debug('Found cuda version: {}'.format(cuda_version))
        return cuda_version

    logger.debug('Could not determine cuda version from "{}"'.format(nvcc_version))
    return None


def probe_cpu_variant(mxnet_features: Dict[str, bool]) -> str:
    """
    Returns the mxnet cpu targeted variant depending on which mxnet features are enabled
    :param mxnet_features: An mxnet feature dictionary of feature to boolean (True = enabled)
    :return: Either cpu, or mkl as the variant
    """
    logger.debug('Determining cpu variant')
    if mxnet_features['MKLDNN']:
        logger.debug('variant is: mkl')
        return 'mkl'

    logger.debug('variant is: cpu')
    return 'cpu'


def probe_gpu_variant(mxnet_features: Dict[str, bool]) -> Optional[str]:
    """
    Returns the mxnet gpu variant depending on which mxnet features are enabled
    :param mxnet_features: An mxnet feature dictionary of feature to boolean (True = enabled)
    :return: The mxnet gpu variant, eg. cu90, cu90mkl, etc.
    :raises RuntimeError is the CUDA feature is not enabled in the library
    """
    if not mxnet_features['CUDA']:
        raise RuntimeError('Cannot determine gpu variant. CUDA feature is disabled.')

    cuda_version = get_cuda_version()
    if cuda_version:
        variant = 'cu{}'.format(cuda_version)
        if mxnet_features['MKLDNN']:
            variant = '{}mkl'.format(variant)
        logger.debug('variant is: {}'.format(variant))
        return variant

    raise RuntimeError('Error determining mxnet variant: Could not retrieve cuda version')


def probe_mxnet_variant(limxnet_path: str) -> Optional[str]:
    """
    Probes the libmxnet library and environment to determine
    the mxnet variant, eg. cpu, mkl, cu90, cu90mkl, etc.
    :return:
    """
    logger.debug('Probing for mxnet variant')
    features = get_libmxnet_features(limxnet_path)
    if not features:
        logger.debug('Error: could not determine variant. Features could not be extracted from libmxnet')
        return None

    if features['CUDA']:
        return probe_gpu_variant(features)
    return probe_cpu_variant(features)


def probe_artifact_repository_bucket() -> Optional[str]:
    """
    Probes environment variables in search of artifact repository bucket
    :return: string containing the artifact repository bucket name
    """
    logger.debug('Probing for artifact repository bucket name')
    bucket = os.environ.get('ARTIFACT_REPOSITORY_BUCKET')
    if not bucket:
        logger.debug('ARTIFACT_REPOSITORY_BUCKET environment variable not found')
    return bucket


def probe(args: argparse.Namespace) -> argparse.Namespace:
    """
    Probes the system to set any arguments that weren't manually set.
    Modifies the input Namespace object with the probed parameters.
    :param args: The namespace object given by argparse.parse()
    """
    logger.debug('Trying to auto-determine arguments from environment')
    if not args.git_sha:
        commit_id = probe_commit_id()
        if not commit_id:
            logger.error('Could not determine commit id. '
                         'Please set it manually with --git-sha, or ensure you are in a cloned '
                         'mxnet repository directory')
            sys.exit(1)
        args.git_sha = commit_id

    if not args.variant:
        variant = probe_mxnet_variant(args.libmxnet)
        if not variant:
            logger.error('Could not determine mxnet variant. Please set it manually with --variant')
            sys.exit(1)
        args.variant = variant

    if not args.os:
        operating_system = probe_operating_system()
        if not operating_system:
            logger.error('Could not determine operating system. Please set it manually with --os')
            sys.exit(1)
        args.os = operating_system

    if not args.bucket:
        artifact_repo_bucket = probe_artifact_repository_bucket()
        if not artifact_repo_bucket:
            logger.error('Could not determine artifact repository bucket. Please set it manually with --bucket')
            sys.exit(1)
        args.bucket = artifact_repo_bucket

    return args


def get_s3_key_prefix(args: argparse.Namespace, subdir: str = '') -> str:
    """
    Returns the S3 key prefix given the arguments namespace
    :param args: The arguments passed in by the user or derived by the script
    :param subdir: An optional subdirectory in which to store the files. Post-pended to the end of the prefix.
    :return: A string containing the S3 key prefix to be used to uploading and downloading files to the artifact repository
    """
    prefix = "{git_sha}/{libtype}/{os}/{variant}/".format(**vars(args))
    if subdir:
        return "{}{}/".format(prefix, subdir)
    return prefix


def push_artifact(args: argparse.Namespace):
    """
    Pushes the artifact to the artifact repository
    :param args: The arguments passed in to this script by the user
    :return 0 for success, non-zero for failure
    """

    args = probe(args)

    logger.info('Pushing artifact with: ')
    logger.info('COMMIT ID   : {}'.format(args.git_sha))
    logger.info('OS          : {}'.format(args.os))
    logger.info('VARIANT     : {}'.format(args.variant))
    logger.info("LIBMXNET    : {}".format(args.libmxnet))
    logger.info("LICENSES    : {}".format(args.licenses))
    logger.info("DEPENDENCIES: {}".format(args.dependencies))
    logger.info("")

    if not args.licenses:
        raise RuntimeError('No licenses defined. Please submit the licenses to be shipped with the binary.')

    # Upload mxnet
    try:
        logger.info('Uploading libmxnet library...')
        s3_upload(args.bucket, get_s3_key_prefix(args), [args.libmxnet])
        logger.info("")

        # Upload licenses
        logger.info('Uploading licenses...')
        s3_upload(args.bucket, get_s3_key_prefix(args, subdir='licenses'), args.licenses)
        logger.info("")

        # Upload dependencies, if necessary
        if args.dependencies:
            logger.info('Uploading dependencies...')
            s3_upload(args.bucket, get_s3_key_prefix(args, subdir='dependencies'), args.dependencies)
            logger.info("")
    except botocore.exceptions.BotoCoreError as e:
        logger.error('Error uploading artifact')
        logger.error(e)
        raise e

    logger.info('Successfully pushed artifact')


def pull_artifact(args: argparse.Namespace):
    """
    Pulls the artifact from the artifact repository
    :param args: The arguments passed in to this script by the user
    :return 0 for success, 1 for unexpected failure, 2 for no artifact found failure
    """
    if not args.variant:
        logger.warning('''variant not set. Using 'cpu' by default.''')
        args.variant = 'cpu'

    args = probe(args)

    logger.info('Pulling artifact with: ')
    logger.info('COMMIT ID   : {}'.format(args.git_sha))
    logger.info('OS          : {}'.format(args.os))
    logger.info('VARIANT     : {}'.format(args.variant))
    logger.info('To directory: {}'.format(args.destination))

    try:
        if not try_s3_download(args.bucket, get_s3_key_prefix(args), args.destination):
            raise RuntimeError('No artifacts found for this configuration.')
        write_libmxnet_meta(args=args, destination=args.destination)
    except botocore.exceptions.BotoCoreError as e:
        logger.error('Error downloading artifact')
        logger.error(e)
        raise e

    logger.info('Successfully pulled artifact')


def is_file(path: str) -> str:
    """
    Returns true or false if path points to an existing file
    :param path: A path to a file
    :return: True if file exists and is a file, False otherwise
    :raises FileNotFoundError if file does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError('''File '{}' not found'''.format(path))
    return os.path.isfile(path)


def sanitize_path_array(paths: List[str]) -> List[str]:
    """
    Expands supplied paths and removes empty or non-file entries.
    :param paths: A list of paths
    :return: A sanitized list of paths
    :raises FileNotFoundError if a file does not exist
    """
    expanded_paths = list(chain(*[glob.glob(path.strip()) for path in paths if path.strip() != '']))
    return [path.strip() for path in expanded_paths if path.strip() != '' and is_file(path)]


def main() -> int:
    config_logging()

    logger.info("MXNet-CD Artifact Repository Tool")

    parser = argparse.ArgumentParser(description="Utility for uploading and downloading MXNet artifacts")

    parser.add_argument("--push",
                        help="Upload artifact to repository",
                        required=False,
                        action='store_true')

    parser.add_argument("--pull",
                        help="Download artifact from repository",
                        required=False,
                        action='store_true')

    parser.add_argument("--libmxnet",
                        help="Path to libmxnet library",
                        required=False,
                        type=str)

    parser.add_argument("--licenses",
                        help="Paths to license files",
                        required=False,
                        nargs=argparse.ZERO_OR_MORE,
                        default=[])

    parser.add_argument("--dependencies",
                        help="Paths to dependencies",
                        required=False,
                        nargs=argparse.ZERO_OR_MORE,
                        default=[])

    parser.add_argument("--os",
                        help="Target operating system",
                        type=str)

    parser.add_argument("--git-sha",
                        help="MXNet repository commit id",
                        required=False,
                        type=str)

    parser.add_argument("--variant",
                        help="MXNet binary variant. Eg. cpu, mkl, cu90, cu100mkl, etc.",
                        required=False,
                        type=str)

    parser.add_argument("--libtype",
                        help="libmxnet dependency linking type",
                        choices=['static', 'dynamic'],
                        default='dynamic',
                        required=False)

    parser.add_argument('--bucket',
                        help="S3 bucket to store files",
                        type=str,
                        required=False)

    parser.add_argument('--destination',
                        help="Destination for downloaded library and supporting files",
                        type=str,
                        default=os.getcwd(),
                        required=False)

    parser.add_argument('--verbose',
                        help='Verbose',
                        action='store_true',
                        default=False)

    parser.add_argument('--debug',
                        help='Debug mode',
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.debug:
        # Set debug level on root logger (ie. for all other loggers)
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.push and not args.pull:
        logger.info('''Mode not specified. Using 'push' by default.''')
        args.push = True

    # libmxnet argument is required for push mode
    if args.push and not args.libmxnet:
        logger.error('Path to libmxnet library must be specified when in push mode. '
                     'Please specify it with --libmxnet.')
        return 1

    # sanitize license and dependency arrays
    # Remove empty or directory entries
    args.licenses = sanitize_path_array(args.licenses)
    args.dependencies = sanitize_path_array(args.dependencies)

    # expand destination path
    args.destination = os.path.abspath(args.destination)

    try:
        if args.push:
            push_artifact(args)

        elif args.pull:
            pull_artifact(args)
    except RuntimeError as err:
        logger.error(err)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
