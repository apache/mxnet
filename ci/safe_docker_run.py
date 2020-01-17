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
Docker command wrapper to guard against Zombie containers
"""

import argparse
import atexit
import logging
import os
import signal
import sys
from functools import reduce
from itertools import chain
from typing import Dict, Any

import docker
from docker.errors import NotFound
from docker.models.containers import Container

from util import config_logging

DOCKER_STOP_TIMEOUT_SECONDS = 3
CONTAINER_WAIT_SECONDS = 600


class SafeDockerClient:
    """
    A wrapper around the docker client to ensure that no zombie containers are left hanging around
    in case the script is not allowed to finish normally
    """

    @staticmethod
    def _trim_container_id(cid):
        """:return: trimmed container id"""
        return cid[:12]

    def __init__(self):
        self._docker_client = docker.from_env()
        self._containers = set()
        self._docker_stop_timeout = DOCKER_STOP_TIMEOUT_SECONDS
        self._container_wait_seconds = CONTAINER_WAIT_SECONDS

        def signal_handler(signum, _):
            signal.pthread_sigmask(signal.SIG_BLOCK, {signum})
            logging.warning("Signal %d received, cleaning up...", signum)
            self._clean_up()
            logging.warning("done. Exiting with error.")
            sys.exit(1)

        atexit.register(self._clean_up)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def _clean_up(self):
        if self._containers:
            logging.warning("Cleaning up containers")
        else:
            return
        # noinspection PyBroadException
        try:
            stop_timeout = int(os.environ.get("DOCKER_STOP_TIMEOUT", self._docker_stop_timeout))
        except Exception:
            stop_timeout = 3
        for container in self._containers:
            try:
                container.stop(timeout=stop_timeout)
                logging.info("☠: stopped container %s", self._trim_container_id(container.id))
                container.remove()
                logging.info("🚽: removed container %s", self._trim_container_id(container.id))
            except Exception as e:
                logging.exception(e)
        self._containers.clear()
        logging.info("Cleaning up containers finished.")

    def _add_container(self, container: Container) -> Container:
        self._containers.add(container)
        return container

    def _remove_container(self, container: Container):
        self._containers.remove(container)

    def run(self, *args, **kwargs) -> int:
        if "detach" in kwargs and kwargs.get("detach") is False:
            raise ValueError("Can only safe run with 'detach' set to True")
        else:
            kwargs["detach"] = True

        # These variables are passed to the container so the process tree killer can find runaway
        # process inside the container
        # https://wiki.jenkins.io/display/JENKINS/ProcessTreeKiller
        # https://github.com/jenkinsci/jenkins/blob/578d6bacb33a5e99f149de504c80275796f0b231/core/src/main/java/hudson/model/Run.java#L2393
        if "environment" not in kwargs:
            kwargs["environment"] = {}

        jenkins_env_vars = ["BUILD_NUMBER", "BUILD_ID", "BUILD_TAG"]
        kwargs["environment"].update({k: os.environ[k] for k in jenkins_env_vars if k in os.environ})

        ret = 0
        try:
            # Race condition:
            # If the call to docker_client.containers.run is interrupted, it is possible that
            # the container won't be cleaned up. We avoid this by temporarily masking the signals.
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            container = self._add_container(self._docker_client.containers.run(*args, **kwargs))
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
            logging.info("Started container: %s", self._trim_container_id(container.id))
            stream = container.logs(stream=True, stdout=True, stderr=True)
            sys.stdout.flush()
            for chunk in stream:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            sys.stdout.flush()
            stream.close()

            try:
                logging.info("Waiting for status of container %s for %d s.",
                             self._trim_container_id(container.id),
                             self._container_wait_seconds)
                wait_result = container.wait(timeout=self._container_wait_seconds)
                logging.info("Container exit status: %s", wait_result)
                ret = wait_result.get('StatusCode', 200)
                if ret != 0:
                    logging.error("Container exited with an error 😞")
                    logging.info("Executed command for reproduction:\n\n%s\n", " ".join(sys.argv))
                else:
                    logging.info("Container exited with success 👍")
                    logging.info("Executed command for reproduction:\n\n%s\n", " ".join(sys.argv))
            except Exception as err:
                logging.exception(err)
                return 150

            try:
                logging.info("Stopping container: %s", self._trim_container_id(container.id))
                container.stop()
            except Exception as e:
                logging.exception(e)
                ret = 151

            try:
                logging.info("Removing container: %s", self._trim_container_id(container.id))
                container.remove()
            except Exception as e:
                logging.exception(e)
                ret = 152
            self._remove_container(container)
            containers = self._docker_client.containers.list()
            if containers:
                logging.info("Other running containers: %s", [self._trim_container_id(x.id) for x in containers])
        except NotFound as e:
            logging.info("Container was stopped before cleanup started: %s", e)

        return ret


def _volume_mount(volume_dfn: str) -> Dict[str, Any]:
    """
    Converts docker volume mount format, e.g. docker run --volume /local/path:/container/path:ro
    to an object understood by the python docker library, e.g. {"local/path": {"bind": "/container/path", "mode": "ro"}}
    This is used by the argparser for automatic conversion and input validation.
    If the mode is not specified, 'rw' is assumed.
    :param volume_dfn: A string to convert to a volume mount object in the format <local path>:<container path>[:ro|rw]
    :return: An object in the form {"<local path>" : {"bind": "<container path>", "mode": "rw|ro"}}
    """
    if volume_dfn is None:
        raise argparse.ArgumentTypeError("Missing value for volume definition")

    parts = volume_dfn.split(":")

    if len(parts) < 2 or len(parts) > 3:
        raise argparse.ArgumentTypeError("Invalid volume definition {}".format(volume_dfn))

    mode = "rw"
    if len(parts) == 3:
        mode = parts[2]

    if mode not in ["rw", "ro"]:
        raise argparse.ArgumentTypeError("Invalid volume mount mode {} in volume definition {}".format(mode, volume_dfn))

    return {parts[0]: {"bind": parts[1], "mode": mode}}


def main(command_line_arguments):
    config_logging()

    parser = argparse.ArgumentParser(
        description="""Wrapper around docker run that protects against Zombie containers""", epilog="")

    parser.add_argument("-u", "--user",
                        help="Username or UID (format: <name|uid>[:<group|gid>])",
                        default=None)

    parser.add_argument("-v", "--volume",
                        action='append',
                        type=_volume_mount,
                        help="Bind mount a volume",
                        default=[])

    parser.add_argument("--cap-add",
                        help="Add Linux capabilities",
                        action="append",
                        type=str,
                        default=[])

    parser.add_argument("--runtime",
                        help="Runtime to use for this container",
                        default=None)

    parser.add_argument("--name",
                        help="Assign a name to the container",
                        default=None)

    parser.add_argument("image", metavar="IMAGE")
    parser.add_argument("command", metavar="COMMAND")
    parser.add_argument("args", nargs='*', metavar="ARG")

    args = parser.parse_args(args=command_line_arguments)
    docker_client = SafeDockerClient()
    return docker_client.run(args.image, **{
        "command": " ".join(list(chain([args.command] + args.args))),
        "user": args.user,
        "runtime": args.runtime,
        "name": args.name,
        "volumes": reduce(lambda dct, v: {**dct, **v}, args.volume, {}),
        "cap_add": args.cap_add
    })


if __name__ == "__main__":
    exit(main(sys.argv[1:]))
