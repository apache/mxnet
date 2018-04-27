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

import argparse
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from itertools import chain
from subprocess import call, check_call
from typing import *


def get_platforms(path: Optional[str]="docker"):
    """Get a list of architectures given our dockerfiles"""
    dockerfiles = glob.glob(os.path.join(path, "Dockerfile.build.*"))
    dockerfiles = list(filter(lambda x: x[-1] != '~', dockerfiles))
    files = list(map(lambda x: re.sub(r"Dockerfile.build.(.*)", r"\1", x), dockerfiles))
    platforms = list(map(lambda x: os.path.split(x)[1], sorted(files)))
    return platforms


def get_docker_tag(platform: str) -> str:
    return "mxnet/build.{0}".format(platform)


def get_dockerfile(platform: str, path="docker") -> str:
    return os.path.join(path, "Dockerfile.build.{0}".format(platform))


def get_docker_binary(use_nvidia_docker: bool) -> str:
    return "nvidia-docker" if use_nvidia_docker else "docker"


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


def buildir() -> str:
    return os.path.join(get_mxnet_root(), "build")


def container_run(platform: str,
                  docker_binary: str,
                  shared_memory_size: str,
                  command: List[str],
                  dry_run: bool = False,
                  into_container: bool = False) -> str:
    tag = get_docker_tag(platform)
    mx_root = get_mxnet_root()
    local_build_folder = buildir()
    # We need to create it first, otherwise it will be created by the docker daemon with root only permissions
    os.makedirs(local_build_folder, exist_ok=True)
    runlist = [docker_binary, 'run', '--rm', '-t',
        '--shm-size={}'.format(shared_memory_size),
        '-v', "{}:/work/mxnet".format(mx_root), # mount mxnet root
        '-v', "{}:/work/build".format(local_build_folder), # mount mxnet/build for storing build artifacts
        '-u', '{}:{}'.format(os.getuid(), os.getgid()),
        tag]
    runlist.extend(command)
    cmd = ' '.join(runlist)
    if not dry_run and not into_container:
        logging.info("Running %s in container %s", command, tag)
        logging.info("Executing: %s", cmd)
        ret = call(runlist)

    into_cmd = deepcopy(runlist)
    idx = into_cmd.index('-u') + 2
    into_cmd[idx:idx] = ['-ti', '--entrypoint', '/bin/bash']
    docker_run_cmd = ' '.join(into_cmd)
    if not dry_run and into_container:
        check_call(into_cmd)

    if not dry_run and ret != 0:
        logging.error("Running of command in container failed (%s): %s", ret, cmd)
        logging.error("You can try to get into the container by using the following command: %s", docker_run_cmd)
        raise subprocess.CalledProcessError(ret, cmd)

    return docker_run_cmd


def list_platforms() -> str:
    print("\nSupported platforms:\n{}".format('\n'.join(get_platforms())))


def main() -> int:
    # We need to be in the same directory than the script so the commands in the dockerfiles work as
    # expected. But the script can be invoked from a different path
    base = os.path.split(os.path.realpath(__file__))[0]
    os.chdir(base)

    logging.getLogger().setLevel(logging.INFO)

    def script_name() -> str:
        return os.path.split(sys.argv[0])[1]

    logging.basicConfig(format='{}: %(asctime)-15s %(message)s'.format(script_name()))

    parser = argparse.ArgumentParser(description="""Utility for building and testing MXNet on docker
    containers""",epilog="")
    parser.add_argument("-p", "--platform",
                        help="platform",
                        type=str)

    parser.add_argument("--build-only",
                        help="Only build the container, don't build the project",
                        action='store_true')

    parser.add_argument("-a", "--all",
                        help="build for all platforms",
                        action='store_true')

    parser.add_argument("-n", "--nvidiadocker",
                        help="Use nvidia docker",
                        action='store_true')

    parser.add_argument("--shm-size",
                        help="Size of the shared memory /dev/shm allocated in the container (e.g '1g')",
                        default='500m',
                        dest="shared_memory_size")

    parser.add_argument("-l", "--list",
                        help="List platforms",
                        action='store_true')

    parser.add_argument("--print-docker-run",
                        help="print docker run command for manual inspection",
                        action='store_true')

    parser.add_argument("-i", "--into-container",
                        help="go in a shell inside the container",
                        action='store_true')

    parser.add_argument("command",
                        help="command to run in the container",
                        nargs='*', action='append', type=str)

    args = parser.parse_args()
    command = list(chain(*args.command))
    docker_binary = get_docker_binary(args.nvidiadocker)
    shared_memory_size = args.shared_memory_size

    print("into container: {}".format(args.into_container))
    if args.list:
        list_platforms()
    elif args.platform:
        platform = args.platform
        build_docker(platform, docker_binary)
        if args.build_only:
            logging.warn("Container was just built. Exiting due to build-only.")
            return 0

        tag = get_docker_tag(platform)
        if command:
            container_run(platform, docker_binary, shared_memory_size, command)
        elif args.print_docker_run:
            print(container_run(platform, docker_binary, shared_memory_size, [], True))
        elif args.into_container:
            container_run(platform, docker_binary, shared_memory_size, [], False, True)
        else:
            cmd = ["/work/mxnet/ci/docker/runtime_functions.sh", "build_{}".format(platform)]
            logging.info("No command specified, trying default build: %s", ' '.join(cmd))
            container_run(platform, docker_binary, shared_memory_size, cmd)

    elif args.all:
        platforms = get_platforms()
        logging.info("Building for all architectures: {}".format(platforms))
        logging.info("Artifacts will be produced in the build/ directory.")
        for platform in platforms:
            build_docker(platform, docker_binary)
            if args.build_only:
                continue
            build_platform = "build_{}".format(platform)
            cmd = ["/work/mxnet/ci/docker/runtime_functions.sh", build_platform]
            shutil.rmtree(buildir(), ignore_errors=True)
            container_run(platform, docker_binary, shared_memory_size, cmd)
            plat_buildir = os.path.join(get_mxnet_root(), build_platform)
            shutil.move(buildir(), plat_buildir)
            logging.info("Built files left in: %s", plat_buildir)

    else:
        parser.print_help()
        list_platforms()
        print("""
Examples:

./build.py -p armv7

    Will build a docker container with cross compilation tools and build MXNet for armv7 by
    running: ci/docker/runtime_functions.sh build_armv7 inside the container.

./build.py -p armv7 ls

    Will execute the given command inside the armv7 container

./build.py -p armv7 --print-docker-run

    Will print a docker run command to get inside the container in an interactive shell

./build.py -p armv7 --into-container

    Will execute a shell into the container

./build.py -a

    Builds for all platforms and leaves artifacts in build_<platform>

    """)

    return 0


if __name__ == '__main__':
    sys.exit(main())
