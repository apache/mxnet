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

"""Multi arch dockerized build tool.

"""

__author__ = 'Marco de Abreu, Kellen Sunderland, Anton Chernov, Pedro Larroy'
__version__ = '0.1'

import os
import sys
import subprocess
import logging
import argparse
from subprocess import check_call, call
import glob
import re
from typing import *
from itertools import chain
from copy import deepcopy


def get_platforms(path: Optional[str]="docker"):
    """Get a list of architectures given our dockerfiles"""
    dockerfiles = glob.glob(os.path.join(path, "Dockerfile.build.*"))
    dockerfiles = list(filter(lambda x: x[-1] != '~', dockerfiles))
    files = list(map(lambda x: re.sub(r"Dockerfile.build.(.*)", r"\1", x), dockerfiles))
    files.sort()
    platforms = list(map(lambda x: os.path.split(x)[1], files))
    return platforms


def get_docker_tag(platform: str) -> None:
    return "mxnet/build.{0}".format(platform)


def get_dockerfile(platform: str, path="docker"):
    return os.path.join(path, "Dockerfile.build.{0}".format(platform))

def get_docker_binary(use_nvidia_docker: bool):
    if use_nvidia_docker:
        return "nvidia-docker"
    else:
        return "docker"

def build_docker(platform: str, docker_binary: str) -> None:
    """Build a container for the given platform"""
    tag = get_docker_tag(platform)
    logging.info("Building container tagged '%s' with %s", tag, docker_binary)
    cmd = [docker_binary, "build",
        "-f", get_dockerfile(platform),
        "--build-arg", "USER_ID={}".format(os.getuid()),
        "-t", tag,
        "docker"]
    logging.info("Running command: '%s'", ' '.join(cmd))
    check_call(cmd)

def get_mxnet_root() -> str:
    curpath = os.path.abspath(os.path.dirname(__file__))
    def is_mxnet_root(path: str) -> bool:
        return os.path.exists(os.path.join(path, ".mxnet_root"))
    while not is_mxnet_root(curpath):
        parent = os.path.abspath(os.path.join(curpath, os.pardir))
        if parent == curpath:
            raise RuntimeError("Got to the root and couldn't find a parent folder with .mxnet_root")
        curpath = parent
    return curpath


def container_run(platform: str, docker_binary: str, command: List[str]) -> None:
    tag = get_docker_tag(platform)
    mx_root = get_mxnet_root()
    local_build_folder = '{}/build'.format(mx_root)
    # We need to create it first, otherwise it will be created by the docker daemon with root only permissions
    os.makedirs(local_build_folder, exist_ok=True)
    logging.info("Running %s in container %s", command, tag)
    runlist = [docker_binary, 'run', '--rm',
        '-v', "{}:/work/mxnet".format(mx_root), # mount mxnet root
        '-v', "{}:/work/build".format(local_build_folder), # mount mxnet/build for storing build artifacts
        '-u', '{}:{}'.format(os.getuid(), os.getgid()),
        tag]
    runlist.extend(command)
    cmd = ' '.join(runlist)
    logging.info("Executing: %s", cmd)
    ret = call(runlist)
    if ret != 0:
        logging.error("Running of command in container failed: %s", cmd)
        into_cmd = deepcopy(runlist)
        idx = into_cmd.index('-u') + 2
        into_cmd[idx:idx] = ['-ti', '--entrypoint', 'bash']
        logging.error("You can try to get into the container by using the following command: %s", ' '.join(into_cmd))
        raise subprocess.CalledProcessError(ret, cmd)

def main() -> int:
    # We need to be in the same directory than the script so the commands in the dockerfiles work as
    # expected. But the script can be invoked from a different path
    base = os.path.split(os.path.realpath(__file__))[0]
    os.chdir(base)

    logging.getLogger().setLevel(logging.INFO)
    def script_name() -> str:
        return os.path.split(sys.argv[0])[1]

    logging.basicConfig(format='{}: %(asctime)-15s %(message)s'.format(script_name()))

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform",
                        help="platform",
                        type=str)

    parser.add_argument("-b", "--build",
                        help="Build the container",
                        action='store_true')

    parser.add_argument("-n", "--nvidiadocker",
                        help="Use nvidia docker",
                        action='store_true')

    parser.add_argument("-l", "--list",
                        help="List platforms",
                        action='store_true')

    parser.add_argument("command",
                        help="command to run in the container",
                        nargs='*', action='append', type=str)

    args = parser.parse_args()
    command = list(chain(*args.command))
    docker_binary = get_docker_binary(args.nvidiadocker)

    if args.list:
        platforms = get_platforms()
        print(platforms)

    elif args.platform:
        platform = args.platform
        if args.build:
            build_docker(platform, docker_binary)
        tag = get_docker_tag(platform)
        if command:
            container_run(platform, docker_binary, command)
        else:
            cmd = ["/work/mxnet/ci/docker/runtime_functions.sh", "build_{}".format(platform)]
            logging.info("No command specified, trying default build: %s", ' '.join(cmd))
            container_run(platform, docker_binary, cmd)

    else:
        platforms = get_platforms()
        logging.info("Building for all architectures: {}".format(platforms))
        logging.info("Artifacts will be produced in the build/ directory.")
        for platform in platforms:
            build_docker(platform, docker_binary)
            cmd = ["/work/mxnet/ci/docker/runtime_functions.sh", "build_{}".format(platform)]
            logging.info("No command specified, trying default build: %s", ' '.join(cmd))
            container_run(platform, docker_binary, cmd)

    return 0


if __name__ == '__main__':
    sys.exit(main())
