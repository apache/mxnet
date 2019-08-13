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
__version__ = '0.3'

import argparse
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from itertools import chain
from subprocess import check_call, check_output
from typing import *
from util import *
import docker
import docker.models
import docker.errors
import signal
import atexit
import pprint


class Cleanup:
    """A class to cleanup containers"""
    def __init__(self):
        self.containers = set()
        self.docker_stop_timeout = 3

    def add_container(self, container: docker.models.containers.Container):
        assert isinstance(container, docker.models.containers.Container)
        self.containers.add(container)

    def remove_container(self, container: docker.models.containers.Container):
        assert isinstance(container, docker.models.containers.Container)
        self.containers.remove(container)

    def _cleanup_containers(self):
        if self.containers:
            logging.warning("Cleaning up containers")
        else:
            return
        # noinspection PyBroadException
        try:
            stop_timeout = int(os.environ.get("DOCKER_STOP_TIMEOUT", self.docker_stop_timeout))
        except Exception:
            stop_timeout = 3
        for container in self.containers:
            try:
                container.stop(timeout=stop_timeout)
                logging.info("☠: stopped container %s", trim_container_id(container.id))
                container.remove()
                logging.info("🚽: removed container %s", trim_container_id(container.id))
            except Exception as e:
                logging.exception(e)
        self.containers.clear()
        logging.info("Cleaning up containers finished.")

    def __call__(self):
        """Perform cleanup"""
        self._cleanup_containers()


def get_dockerfiles_path():
    return "docker"


def get_platforms(path: str = get_dockerfiles_path()) -> List[str]:
    """Get a list of architectures given our dockerfiles"""
    dockerfiles = glob.glob(os.path.join(path, "Dockerfile.*"))
    dockerfiles = list(filter(lambda x: x[-1] != '~', dockerfiles))
    files = list(map(lambda x: re.sub(r"Dockerfile.(.*)", r"\1", x), dockerfiles))
    platforms = list(map(lambda x: os.path.split(x)[1], sorted(files)))
    return platforms


def get_docker_tag(platform: str, registry: str) -> str:
    """:return: docker tag to be used for the container"""
    platform = platform if any(x in platform for x in ['build.', 'publish.']) else 'build.{}'.format(platform)
    if not registry:
        registry = "mxnet_local"
    return "{0}/{1}".format(registry, platform)


def get_dockerfile(platform: str, path=get_dockerfiles_path()) -> str:
    platform = platform if any(x in platform for x in ['build.', 'publish.']) else 'build.{}'.format(platform)
    return os.path.join(path, "Dockerfile.{0}".format(platform))


def get_docker_binary(use_nvidia_docker: bool) -> str:
    return "nvidia-docker" if use_nvidia_docker else "docker"


def build_docker(platform: str, docker_binary: str, registry: str, num_retries: int, no_cache: bool) -> str:
    """
    Build a container for the given platform
    :param platform: Platform
    :param docker_binary: docker binary to use (docker/nvidia-docker)
    :param registry: Dockerhub registry name
    :param num_retries: Number of retries to build the docker image
    :param no_cache: pass no-cache to docker to rebuild the images
    :return: Id of the top level image
    """
    tag = get_docker_tag(platform=platform, registry=registry)
    logging.info("Building docker container tagged '%s' with %s", tag, docker_binary)
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
    # This also prevents using local layers for caching: https://github.com/moby/moby/issues/33002
    # So to use local caching, we should omit the cache-from by using --no-dockerhub-cache argument to this
    # script.
    #
    # This doesn't work with multi head docker files.
    #
    cmd = [docker_binary, "build",
           "-f", get_dockerfile(platform),
           "--build-arg", "USER_ID={}".format(os.getuid()),
           "--build-arg", "GROUP_ID={}".format(os.getgid())]
    if no_cache:
        cmd.append("--no-cache")
    elif registry:
        cmd.extend(["--cache-from", tag])
    cmd.extend(["-t", tag, get_dockerfiles_path()])

    @retry(subprocess.CalledProcessError, tries=num_retries)
    def run_cmd():
        logging.info("Running command: '%s'", ' '.join(cmd))
        check_call(cmd)

    run_cmd()
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
    image_id_b = check_output(cmd)
    image_id = image_id_b.decode('utf-8').strip()
    if not image_id:
        raise RuntimeError('Unable to find docker image id matching with tag {}'.format(docker_tag))
    return image_id


def buildir() -> str:
    return os.path.join(get_mxnet_root(), "build")


def default_ccache_dir() -> str:
    """:return: ccache directory for the current platform"""
    # Share ccache across containers
    if 'CCACHE_DIR' in os.environ:
        ccache_dir = os.path.realpath(os.environ['CCACHE_DIR'])
        try:
            os.makedirs(ccache_dir, exist_ok=True)
            return ccache_dir
        except PermissionError:
            logging.info('Unable to make dirs at %s, falling back to local temp dir', ccache_dir)
    # In osx tmpdir is not mountable by default
    import platform
    if platform.system() == 'Darwin':
        ccache_dir = "/tmp/_mxnet_ccache"
        os.makedirs(ccache_dir, exist_ok=True)
        return ccache_dir
    return os.path.join(os.path.expanduser("~"), ".ccache")


def trim_container_id(cid):
    """:return: trimmed container id"""
    return cid[:12]


def container_run(platform: str,
                  nvidia_runtime: bool,
                  docker_registry: str,
                  shared_memory_size: str,
                  local_ccache_dir: str,
                  command: List[str],
                  cleanup: Cleanup,
                  environment: Dict[str, str],
                  dry_run: bool = False) -> int:
    """Run command in a container"""
    container_wait_s = 600
    #
    # Environment setup
    #
    environment.update({
        'CCACHE_MAXSIZE': '500G',
        'CCACHE_TEMPDIR': '/tmp/ccache',  # temp dir should be local and not shared
        'CCACHE_DIR': '/work/ccache',  # this path is inside the container as /work/ccache is
                                       # mounted
        'CCACHE_LOGFILE': '/tmp/ccache.log',  # a container-scoped log, useful for ccache
                                              # verification.
    })
    # These variables are passed to the container to the process tree killer can find runaway
    # process inside the container
    # https://wiki.jenkins.io/display/JENKINS/ProcessTreeKiller
    # https://github.com/jenkinsci/jenkins/blob/578d6bacb33a5e99f149de504c80275796f0b231/core/src/main/java/hudson/model/Run.java#L2393
    #
    jenkins_env_vars = ['BUILD_NUMBER', 'BUILD_ID', 'BUILD_TAG']
    environment.update({k: os.environ[k] for k in jenkins_env_vars if k in os.environ})
    environment.update({k: os.environ[k] for k in ['CCACHE_MAXSIZE'] if k in os.environ})

    tag = get_docker_tag(platform=platform, registry=docker_registry)
    mx_root = get_mxnet_root()
    local_build_folder = buildir()
    # We need to create it first, otherwise it will be created by the docker daemon with root only permissions
    os.makedirs(local_build_folder, exist_ok=True)
    os.makedirs(local_ccache_dir, exist_ok=True)
    logging.info("Using ccache directory: %s", local_ccache_dir)
    docker_client = docker.from_env()
    # Equivalent command
    docker_cmd_list = [
        get_docker_binary(nvidia_runtime),
        'run',
        "--cap-add",
        "SYS_PTRACE", # Required by ASAN
        '--rm',
        '--shm-size={}'.format(shared_memory_size),
        # mount mxnet root
        '-v', "{}:/work/mxnet".format(mx_root),
        # mount mxnet/build for storing build
        '-v', "{}:/work/build".format(local_build_folder),
        '-v', "{}:/work/ccache".format(local_ccache_dir),
        '-u', '{}:{}'.format(os.getuid(), os.getgid()),
        '-e', 'CCACHE_MAXSIZE={}'.format(environment['CCACHE_MAXSIZE']),
        # temp dir should be local and not shared
        '-e', 'CCACHE_TEMPDIR={}'.format(environment['CCACHE_TEMPDIR']),
        # this path is inside the container as /work/ccache is mounted
        '-e', "CCACHE_DIR={}".format(environment['CCACHE_DIR']),
        # a container-scoped log, useful for ccache verification.
        '-e', "CCACHE_LOGFILE={}".format(environment['CCACHE_LOGFILE']),
        '-ti',
        tag]
    docker_cmd_list.extend(command)
    docker_cmd = ' \\\n\t'.join(docker_cmd_list)
    logging.info("Running %s in container %s", command, tag)
    logging.info("Executing the equivalent of:\n%s\n", docker_cmd)
    # return code of the command inside docker
    ret = 0
    if not dry_run:
        #############################
        #
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        # noinspection PyShadowingNames
        runtime = None
        if nvidia_runtime:
            # noinspection PyShadowingNames
            # runc is default (docker info | grep -i runtime)
            runtime = 'nvidia'
        container = docker_client.containers.run(
            tag,
            runtime=runtime,
            detach=True,
            command=command,
            shm_size=shared_memory_size,
            user='{}:{}'.format(os.getuid(), os.getgid()),
            cap_add='SYS_PTRACE',
            volumes={
                mx_root:
                    {'bind': '/work/mxnet', 'mode': 'rw'},
                local_build_folder:
                    {'bind': '/work/build', 'mode': 'rw'},
                local_ccache_dir:
                    {'bind': '/work/ccache', 'mode': 'rw'},
            },
            environment=environment)
        try:
            logging.info("Started container: %s", trim_container_id(container.id))
            # Race condition:
            # If the previous call is interrupted then it's possible that the container is not cleaned up
            # We avoid by masking the signals temporarily
            cleanup.add_container(container)
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
            #
            #############################

            stream = container.logs(stream=True, stdout=True, stderr=True)
            sys.stdout.flush()
            for chunk in stream:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            sys.stdout.flush()
            stream.close()
            try:
                logging.info("Waiting for status of container %s for %d s.",
                            trim_container_id(container.id),
                            container_wait_s)
                wait_result = container.wait(timeout=container_wait_s)
                logging.info("Container exit status: %s", wait_result)
                ret = wait_result.get('StatusCode', 200)
                if ret != 0:
                    logging.error("Container exited with an error 😞")
                    logging.info("Executed command for reproduction:\n\n%s\n", " ".join(sys.argv))
                else:
                    logging.info("Container exited with success 👍")
            except Exception as e:
                logging.exception(e)
                ret = 150

            # Stop
            try:
                logging.info("Stopping container: %s", trim_container_id(container.id))
                container.stop()
            except Exception as e:
                logging.exception(e)
                ret = 151

            # Remove
            try:
                logging.info("Removing container: %s", trim_container_id(container.id))
                container.remove()
            except Exception as e:
                logging.exception(e)
                ret = 152
            cleanup.remove_container(container)
            containers = docker_client.containers.list()
            if containers:
                logging.info("Other running containers: %s", [trim_container_id(x.id) for x in containers])
        except docker.errors.NotFound as e:
            logging.info("Container was stopped before cleanup started: %s", e)
    return ret


def list_platforms() -> str:
    return "\nSupported platforms:\n{}".format('\n'.join(get_platforms()))


def load_docker_cache(tag, docker_registry) -> None:
    """Imports tagged container from the given docker registry"""
    if docker_registry:
        # noinspection PyBroadException
        try:
            import docker_cache
            logging.info('Docker cache download is enabled from registry %s', docker_registry)
            docker_cache.load_docker_cache(registry=docker_registry, docker_tag=tag)
        except Exception:
            logging.exception('Unable to retrieve Docker cache. Continue without...')
    else:
        logging.info('Distributed docker cache disabled')


def log_environment():
    instance_id = ec2_instance_id_hostname()
    if instance_id:
        logging.info("EC2 Instance id: %s", instance_id)
    pp = pprint.PrettyPrinter(indent=4)
    logging.debug("Build environment: %s", pp.pformat(dict(os.environ)))


def script_name() -> str:
    """:returns: script name with leading paths removed"""
    return os.path.split(sys.argv[0])[1]

def config_logging():
    import time
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.basicConfig(format='{}: %(asctime)sZ %(levelname)s %(message)s'.format(script_name()))
    logging.Formatter.converter = time.gmtime

def main() -> int:
    config_logging()

    logging.info("MXNet container based build tool.")
    log_environment()
    chdir_to_script_directory()

    parser = argparse.ArgumentParser(description="""Utility for building and testing MXNet on docker
    containers""", epilog="")
    parser.add_argument("-p", "--platform",
                        help="platform",
                        type=str)

    parser.add_argument("-b", "--build-only",
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

    parser.add_argument("-d", "--docker-registry",
                        help="Dockerhub registry name to retrieve cache from.",
                        default='mxnetci',
                        type=str)

    parser.add_argument("-r", "--docker-build-retries",
                        help="Number of times to retry building the docker image. Default is 1",
                        default=1,
                        type=int)

    parser.add_argument("--no-cache", action="store_true",
                        help="passes --no-cache to docker build")

    parser.add_argument("-e", "--environment", nargs="*", default=[],
                        help="Environment variables for the docker container. "
                        "Specify with a list containing either names or name=value")

    parser.add_argument("command",
                        help="command to run in the container",
                        nargs='*', action='append', type=str)

    parser.add_argument("--ccache-dir",
                        default=default_ccache_dir(),
                        help="ccache directory",
                        type=str)

    args = parser.parse_args()

    command = list(chain(*args.command))
    docker_binary = get_docker_binary(args.nvidiadocker)

    # Cleanup on signals and exit
    cleanup = Cleanup()

    def signal_handler(signum, _):
        signal.pthread_sigmask(signal.SIG_BLOCK, {signum})
        logging.warning("Signal %d received, cleaning up...", signum)
        cleanup()
        logging.warning("done. Exiting with error.")
        sys.exit(1)

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    environment = dict([(e.split('=')[:2] if '=' in e else (e, os.environ[e]))
                        for e in args.environment])

    if args.list:
        print(list_platforms())
    elif args.platform:
        platform = args.platform
        tag = get_docker_tag(platform=platform, registry=args.docker_registry)
        if args.docker_registry:
            load_docker_cache(tag=tag, docker_registry=args.docker_registry)
        build_docker(platform=platform, docker_binary=docker_binary, registry=args.docker_registry,
                     num_retries=args.docker_build_retries, no_cache=args.no_cache)
        if args.build_only:
            logging.warning("Container was just built. Exiting due to build-only.")
            return 0

        # noinspection PyUnusedLocal
        ret = 0
        if command:
            ret = container_run(
                platform=platform, nvidia_runtime=args.nvidiadocker,
                shared_memory_size=args.shared_memory_size, command=command, docker_registry=args.docker_registry,
                local_ccache_dir=args.ccache_dir, cleanup=cleanup, environment=environment)
        elif args.print_docker_run:
            command = []
            ret = container_run(
                platform=platform, nvidia_runtime=args.nvidiadocker,
                shared_memory_size=args.shared_memory_size, command=command, docker_registry=args.docker_registry,
                local_ccache_dir=args.ccache_dir, dry_run=True, cleanup=cleanup, environment=environment)
        else:
            # With no commands, execute a build function for the target platform
            command = ["/work/mxnet/ci/docker/runtime_functions.sh", "build_{}".format(platform)]
            logging.info("No command specified, trying default build: %s", ' '.join(command))
            ret = container_run(
                platform=platform, nvidia_runtime=args.nvidiadocker,
                shared_memory_size=args.shared_memory_size, command=command, docker_registry=args.docker_registry,
                local_ccache_dir=args.ccache_dir, cleanup=cleanup, environment=environment)

        if ret != 0:
            logging.critical("Execution of %s failed with status: %d", command, ret)
            return ret

    elif args.all:
        platforms = get_platforms()
        platforms = [platform for platform in platforms if 'build.' in platform]
        logging.info("Building for all architectures: %s", platforms)
        logging.info("Artifacts will be produced in the build/ directory.")
        for platform in platforms:
            tag = get_docker_tag(platform=platform, registry=args.docker_registry)
            load_docker_cache(tag=tag, docker_registry=args.docker_registry)
            build_docker(platform, docker_binary=docker_binary, registry=args.docker_registry,
                         num_retries=args.docker_build_retries, no_cache=args.no_cache)
            if args.build_only:
                continue
            shutil.rmtree(buildir(), ignore_errors=True)
            build_platform = "build_{}".format(platform)
            plat_buildir = os.path.abspath(os.path.join(get_mxnet_root(), '..',
                                                        "mxnet_{}".format(build_platform)))
            if os.path.exists(plat_buildir):
                logging.warning("%s already exists, skipping", plat_buildir)
                continue
            command = ["/work/mxnet/ci/docker/runtime_functions.sh", build_platform]
            container_run(
                platform=platform, nvidia_runtime=args.nvidiadocker,
                shared_memory_size=args.shared_memory_size, command=command, docker_registry=args.docker_registry,
                local_ccache_dir=args.ccache_dir, cleanup=cleanup, environment=environment)
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

./build.py -a

    Builds for all platforms and leaves artifacts in build_<platform>

    """)

    return 0


if __name__ == '__main__':
    sys.exit(main())
