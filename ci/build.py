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
import tempfile
import platform
from copy import deepcopy
from itertools import chain
from subprocess import call, check_call
from typing import *
from util import *

CCACHE_MAXSIZE = '500G'

def under_ci() -> bool:
    """:return: True if we run in Jenkins."""
    return 'JOB_NAME' in os.environ

def get_platforms(path: Optional[str] = "docker"):
    """Get a list of architectures given our dockerfiles"""
    dockerfiles = glob.glob(os.path.join(path, "Dockerfile.build.*"))
    dockerfiles = list(filter(lambda x: x[-1] != '~', dockerfiles))
    files = list(map(lambda x: re.sub(r"Dockerfile.build.(.*)", r"\1", x), dockerfiles))
    platforms = list(map(lambda x: os.path.split(x)[1], sorted(files)))
    return platforms


def get_docker_tag(platform: str, registry: str) -> str:
    return "{0}/build.{1}".format(registry, platform)


def get_dockerfile(platform: str, path="docker") -> str:
    return os.path.join(path, "Dockerfile.build.{0}".format(platform))


def get_docker_binary(use_nvidia_docker: bool) -> str:
    return "nvidia-docker" if use_nvidia_docker else "docker"


def build_docker(platform: str, docker_binary: str, registry: str, num_retries: int) -> None:
    """
    Build a container for the given platform
    :param platform: Platform
    :param docker_binary: docker binary to use (docker/nvidia-docker)
    :param registry: Dockerhub registry name
    :param num_retries: Number of retries to build the docker image
    :return: Id of the top level image
    """

    tag = get_docker_tag(platform=platform, registry=registry)
    logging.info("Building container tagged '%s' with %s", tag, docker_binary)
    #
    # We add a user with the same group as the executing non-root user so files created in the
    # container match permissions of the local user. Same for the group.
    #
    # These variables are used in the docker files to create user and group with these ids.
    # see: docker/install/ubuntu_adduser.sh
    #
    # cache-from is needed so we use the cached images tagged from the remote via
    # docker pull see: docker_cache.load_docker_cache
    #
    # This doesn't work with multi head docker files.
    # 

    for i in range(num_retries):
        logging.info('%d out of %d tries to build the docker image.', i + 1, num_retries)

        cmd = [docker_binary, "build",
               "-f", get_dockerfile(platform),
               "--build-arg", "USER_ID={}".format(os.getuid()),
               "--build-arg", "GROUP_ID={}".format(os.getgid()),
               "--cache-from", tag,
               "-t", tag,
               "docker"]
        logging.info("Running command: '%s'", ' '.join(cmd))
        try:
            check_call(cmd)
            # Docker build was successful. Call break to break out of the retry mechanism
            break
        except subprocess.CalledProcessError as e:
            saved_exception = e
            logging.error('Failed to build docker image')
            # Building the docker image failed. Call continue to trigger the retry mechanism
            continue
    else:
        # Num retries exceeded
        logging.exception('Exception during build of docker image', saved_exception)
        logging.fatal('Failed to build the docker image, aborting...')
        sys.exit(1)

    # Get image id by reading the tag. It's guaranteed (except race condition) that the tag exists. Otherwise, the
    # check_call would have failed
    image_id = _get_local_image_id(docker_binary=docker_binary, docker_tag=tag)
    if not image_id:
        raise FileNotFoundError('Unable to find docker image id matching with {}'.format(tag))
    return image_id


def _get_local_image_id(docker_binary, docker_tag):
    """
    Get the image id of the local docker layer with the passed tag
    :param docker_tag: docker tag
    :return: Image id as string or None if tag does not exist
    """
    cmd = [docker_binary, "images", "-q", docker_tag]
    image_id_b = subprocess.check_output(cmd)
    image_id = image_id_b.decode('utf-8').strip()
    return image_id


def buildir() -> str:
    return os.path.join(get_mxnet_root(), "build")

def default_ccache_dir() -> str:
    # Share ccache across containers
    if 'CCACHE_DIR' in os.environ:
        try:
            ccache_dir = os.path.realpath(os.environ['CCACHE_DIR'])
            os.makedirs(ccache_dir, exist_ok=True)
            return ccache_dir
        except PermissionError:
            logging.info('Unable to make dirs at %s, falling back to local temp dir', ccache_dir)
    # In osx tmpdir is not mountable by default
    if platform.system() == 'Darwin':
        ccache_dir = "/tmp/_mxnet_ccache"
        os.makedirs(ccache_dir, exist_ok=True)
        return ccache_dir
    return os.path.join(tempfile.gettempdir(), "ci_ccache")


def container_run(platform: str,
                  docker_binary: str,
                  docker_registry: str,
                  shared_memory_size: str,
                  local_ccache_dir: str,
                  command: List[str],
                  dry_run: bool = False,
                  interactive: bool = False) -> str:
    tag = get_docker_tag(platform=platform, registry=docker_registry)
    mx_root = get_mxnet_root()
    local_build_folder = buildir()
    # We need to create it first, otherwise it will be created by the docker daemon with root only permissions
    os.makedirs(local_build_folder, exist_ok=True)
    os.makedirs(local_ccache_dir, exist_ok=True)
    logging.info("Using ccache directory: %s", local_ccache_dir)
    runlist = [docker_binary, 'run', '--rm', '-t',
               '--shm-size={}'.format(shared_memory_size),
               '-v', "{}:/work/mxnet".format(mx_root),  # mount mxnet root
               '-v', "{}:/work/build".format(local_build_folder),  # mount mxnet/build for storing build artifacts
               '-v', "{}:/work/ccache".format(local_ccache_dir),
               '-u', '{}:{}'.format(os.getuid(), os.getgid()),
               '-e', 'CCACHE_MAXSIZE={}'.format(CCACHE_MAXSIZE),
               '-e', 'CCACHE_TEMPDIR=/tmp/ccache',  # temp dir should be local and not shared
               '-e', "CCACHE_DIR=/work/ccache",  # this path is inside the container as /work/ccache is mounted
               '-e', "CCACHE_LOGFILE=/tmp/ccache.log",  # a container-scoped log, useful for ccache verification.
               tag]
    runlist.extend(command)
    cmd = '\\\n\t'.join(runlist)
    ret = 0
    if not dry_run and not interactive:
        logging.info("Running %s in container %s", command, tag)
        logging.info("Executing:\n%s\n", cmd)
        ret = call(runlist)

    docker_run_cmd = ' '.join(runlist)
    if not dry_run and interactive:
        into_cmd = deepcopy(runlist)
        # -ti can't be after the tag, as is interpreted as a command so hook it up after the -u argument
        idx = into_cmd.index('-u') + 2
        into_cmd[idx:idx] = ['-ti']
        cmd = '\\\n\t'.join(into_cmd)
        logging.info("Executing:\n%s\n", cmd)
        docker_run_cmd = ' '.join(into_cmd)
        ret = call(into_cmd)

    if not dry_run and not interactive and ret != 0:
        logging.error("Running of command in container failed (%s):\n%s\n", ret, cmd)
        logging.error("You can get into the container by adding the -i option")
        raise subprocess.CalledProcessError(ret, cmd)

    return docker_run_cmd


def list_platforms() -> str:
    print("\nSupported platforms:\n{}".format('\n'.join(get_platforms())))

def load_docker_cache(tag, docker_registry) -> None:
    if docker_registry:
        try:
            import docker_cache
            logging.info('Docker cache download is enabled from registry %s', docker_registry)
            docker_cache.load_docker_cache(registry=docker_registry, docker_tag=tag)
        except Exception:
            logging.exception('Unable to retrieve Docker cache. Continue without...')
    else:
        logging.info('Distributed docker cache disabled')

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
    containers""", epilog="")
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

    parser.add_argument("-i", "--interactive",
                        help="go in a shell inside the container",
                        action='store_true')

    parser.add_argument("-d", "--docker-registry",
                        help="Dockerhub registry name to retrieve cache from. Default is 'mxnetci'",
                        default='mxnetci',
                        type=str)

    parser.add_argument("-r", "--docker-build-retries",
                        help="Number of times to retry building the docker image. Default is 1",
                        default=1,
                        type=int)

    parser.add_argument("-c", "--cache", action="store_true",
                        help="Enable docker registry cache")

    parser.add_argument("command",
                        help="command to run in the container",
                        nargs='*', action='append', type=str)

    parser.add_argument("--ccache-dir",
                        default=default_ccache_dir(),
                        help="Ccache directory",
                        type=str)

    args = parser.parse_args()
    def use_cache():
        return args.cache or under_ci()

    command = list(chain(*args.command))
    docker_binary = get_docker_binary(args.nvidiadocker)
    shared_memory_size = args.shared_memory_size
    num_docker_build_retires = args.docker_build_retries

    if args.list:
        list_platforms()
    elif args.platform:
        platform = args.platform
        tag = get_docker_tag(platform=platform, registry=args.docker_registry)
        if use_cache():
            load_docker_cache(tag=tag, docker_registry=args.docker_registry)
        build_docker(platform, docker_binary, registry=args.docker_registry, num_retries=num_docker_build_retires)
        if args.build_only:
            logging.warning("Container was just built. Exiting due to build-only.")
            return 0

        if command:
            container_run(platform=platform, docker_binary=docker_binary, shared_memory_size=shared_memory_size,
                          command=command, docker_registry=args.docker_registry,
                          local_ccache_dir=args.ccache_dir, interactive=args.interactive)
        elif args.print_docker_run:
            print(container_run(platform=platform, docker_binary=docker_binary, shared_memory_size=shared_memory_size,
                                command=[], dry_run=True, docker_registry=args.docker_registry, local_ccache_dir=args.ccache_dir))
        elif args.interactive:
            container_run(platform=platform, docker_binary=docker_binary, shared_memory_size=shared_memory_size,
                          command=command, docker_registry=args.docker_registry,
                          local_ccache_dir=args.ccache_dir, interactive=args.interactive)

        else:
            # With no commands, execute a build function for the target platform
            assert not args.interactive, "when running with -i must provide a command"
            cmd = ["/work/mxnet/ci/docker/runtime_functions.sh", "build_{}".format(platform)]
            logging.info("No command specified, trying default build: %s", ' '.join(cmd))
            container_run(platform=platform, docker_binary=docker_binary, shared_memory_size=shared_memory_size,
                          command=cmd, docker_registry=args.docker_registry,
                          local_ccache_dir=args.ccache_dir)

    elif args.all:
        platforms = get_platforms()
        logging.info("Building for all architectures: {}".format(platforms))
        logging.info("Artifacts will be produced in the build/ directory.")
        for platform in platforms:
            tag = get_docker_tag(platform=platform, registry=args.docker_registry)
            if use_cache():
                load_docker_cache(tag=tag, docker_registry=args.docker_registry)
            build_docker(platform, docker_binary, args.docker_registry, num_retries=num_docker_build_retires)
            if args.build_only:
                continue
            build_platform = "build_{}".format(platform)
            cmd = ["/work/mxnet/ci/docker/runtime_functions.sh", build_platform]
            shutil.rmtree(buildir(), ignore_errors=True)
            container_run(platform=platform, docker_binary=docker_binary, shared_memory_size=shared_memory_size,
                          command=cmd, docker_registry=args.docker_registry, local_ccache_dir=args.ccache_dir)
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

    Will print a docker run command to get inside the container in a shell

./build.py -p armv7 --interactive

    Will execute a shell into the container

./build.py -a

    Builds for all platforms and leaves artifacts in build_<platform>

    """)

    return 0


if __name__ == '__main__':
    sys.exit(main())
