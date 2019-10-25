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
Safe docker run tests
"""
import itertools
import os
import signal
import unittest
from typing import Optional
from unittest.mock import create_autospec, patch, call

from docker import DockerClient
from docker.models.containers import Container, ContainerCollection

from safe_docker_run import SafeDockerClient, main


def create_mock_container(status_code: int = 0):
    """
    Creates a mock docker container that exits with the specified status code
    """
    mock_container = create_autospec(Container, name="mock_container")
    mock_container.wait.return_value = {
        "StatusCode": status_code
    }
    return mock_container


def create_mock_container_collection(container: Container):
    """
    Creates a mock ContainerCollection that return the supplied container when the 'run' method is called
    """
    mock_container_collection = create_autospec(ContainerCollection, name="mock_collection")
    mock_container_collection.run.return_value = container
    return mock_container_collection


class MockDockerClient:
    """
    A mock DockerClient when docker.from_env is called
    The supplied container will be returned when the client.containers.run method is called
    """
    def __init__(self, container: Container):
        self._mock_client = create_autospec(DockerClient, name="mock_client")
        self._mock_client.containers = create_mock_container_collection(container)
        self._patch = patch("docker.from_env", return_value=self._mock_client)

    def __enter__(self):
        self._patch.start()
        return self._mock_client

    def __exit__(self, _, __, ___):
        self._patch.stop()


class TestSafeDockerRun(unittest.TestCase):

    @patch("safe_docker_run.signal.pthread_sigmask")
    @patch.dict(os.environ, {
        "BUILD_NUMBER": "BUILD_NUMBER_5",
        "BUILD_ID": "BUILD_ID_1",
        "BUILD_TAG": "BUILD_TAG_7"
    })
    def test_run_successful(self, mock_pthread_sigmask):
        """
        Tests successful run
        """
        mock_container = create_mock_container()

        with MockDockerClient(mock_container) as mock_client:
            safe_docker = SafeDockerClient()

            # Check return code is 0
            assert safe_docker.run("image", "command") == 0

            # Check call to container is correct
            assert mock_client.containers.run.call_args_list == [
                call("image", "command", detach=True, environment={
                    "BUILD_NUMBER": "BUILD_NUMBER_5",
                    "BUILD_ID": "BUILD_ID_1",
                    "BUILD_TAG": "BUILD_TAG_7"
                })
            ]

            # Check correct signals are blocked then unblocked
            assert mock_pthread_sigmask.call_args_list == [
                call(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM}),
                call(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
            ]

            # Assert container is stopped and removed
            assert mock_container.stop.call_count == 1
            assert mock_container.remove.call_count == 1
            assert len(safe_docker._containers) == 0

    def test_run_detach(self):
        """
        Tests detach=True is passed to the underlying call by default
        """
        mock_container = create_mock_container()

        # Test detach=True is passed in even if not specified
        with MockDockerClient(mock_container) as mock_client:
            safe_docker = SafeDockerClient()
            assert safe_docker.run("image", "command") == 0
            assert mock_client.containers.run.call_count == 1
            _, kwargs = mock_client.containers.run.call_args
            assert kwargs["detach"] is True

        # Test passing in detach=True does not cause any issues
        with MockDockerClient(mock_container) as mock_client:
            safe_docker = SafeDockerClient()
            assert safe_docker.run("image", "command", detach=True) == 0
            assert mock_client.containers.run.call_count == 1
            _, kwargs = mock_client.containers.run.call_args
            assert kwargs["detach"] is True

        # Test detach=False fails
        with MockDockerClient(mock_container) as mock_client:
            safe_docker = SafeDockerClient()
            with self.assertRaises(ValueError):
                safe_docker.run("image", "command", detach=False)
                assert mock_client.containers.run.call_args_list == []

    def test_jenkins_vars(self):
        """
        Tests jenkins environment variables are appropriately passed to the underlying docker run call
        """
        # NOTE: It's important that these variables are passed to the underlying docker container
        # These variables are passed to the container so the process tree killer can find runaway
        # process inside the container
        # https://wiki.jenkins.io/display/JENKINS/ProcessTreeKiller
        # https://github.com/jenkinsci/jenkins/blob/578d6bacb33a5e99f149de504c80275796f0b231/core/src/main/java/hudson/model/Run.java#L2393

        jenkins_vars = {
            "BUILD_NUMBER": "BUILD_NUMBER_5",
            "BUILD_ID": "BUILD_ID_1",
            "BUILD_TAG": "BUILD_TAG_7"
        }
        mock_container = create_mock_container()

        # Test environment is empty if the jenkins vars are not present
        with MockDockerClient(mock_container) as mock_client:
            safe_docker = SafeDockerClient()
            assert safe_docker.run("image", "command") == 0
            assert mock_client.containers.run.call_count == 1
            _, kwargs = mock_client.containers.run.call_args
            assert kwargs["environment"] == {}

        # Test environment contains jenkins env vars if they are present
        with MockDockerClient(mock_container) as mock_client:
            with patch.dict(os.environ, jenkins_vars):
                safe_docker = SafeDockerClient()
                assert safe_docker.run("image", "command") == 0
                assert mock_client.containers.run.call_count == 1
                _, kwargs = mock_client.containers.run.call_args
                assert kwargs["environment"] == jenkins_vars

        # Test jenkins env vars are added to callers env vars
        user_env = {"key1": "value1", "key2": "value2"}
        with MockDockerClient(mock_container) as mock_client:
            with patch.dict(os.environ, jenkins_vars):
                safe_docker = SafeDockerClient()
                assert safe_docker.run("image", "command", environment=user_env) == 0
                assert mock_client.containers.run.call_count == 1
                _, kwargs = mock_client.containers.run.call_args
                assert kwargs["environment"] == {**jenkins_vars, **user_env}

    def test_run_args_kwargs_passed(self):
        """
        Tests args and kwargs are passed to the container run call
        """
        mock_container = create_mock_container()

        # Test detach=True is passed in even if not specified
        with MockDockerClient(mock_container) as mock_client:
            safe_docker = SafeDockerClient()
            assert safe_docker.run(
                "image",
                "command",
                "another_arg",
                str_param="value",
                bool_param=True,
                none_param=None,
                int_param=5,
                float_param=5.2,
                list_param=["this", "is", "a", "list"],
                map_param={
                    "a": "5",
                    "b": True,
                    "c": 2
                }) == 0
            assert mock_client.containers.run.call_args_list == [
                call(
                    "image",
                    "command",
                    "another_arg",
                    detach=True,
                    environment={},
                    str_param="value",
                    bool_param=True,
                    none_param=None,
                    int_param=5,
                    float_param=5.2,
                    list_param=["this", "is", "a", "list"],
                    map_param={
                        "a": "5",
                        "b": True,
                        "c": 2
                    }
                )
            ]

    def test_container_returns_non_zero_status_code(self):
        """
        Tests non-zero code from container is returned and the container
        is cleaned up
        """
        mock_container = create_mock_container(status_code=10)
        with MockDockerClient(mock_container):
            safe_docker = SafeDockerClient()
            # check return code and that container gets cleaned up
            assert safe_docker.run("image", "command") == 10
            assert mock_container.stop.call_count == 1
            assert mock_container.remove.call_count == 1
            assert len(safe_docker._containers) == 0

    def test_container_wait_raises_returns_150(self):
        """
        Tests 150 is returned if an error is raised when calling container.wait
        """
        mock_container = create_mock_container()
        mock_container.wait.side_effect = RuntimeError("Something bad happened")
        with MockDockerClient(mock_container):
            safe_docker = SafeDockerClient()
            assert safe_docker.run("image", "command") == 150

    def test_container_stop_raises_returns_151(self):
        """
        Tests 151 is returned if an error is raised when calling container.stop
        """
        mock_container = create_mock_container()
        mock_container.stop.side_effect = RuntimeError("Something bad happened")
        with MockDockerClient(mock_container):
            safe_docker = SafeDockerClient()
            assert safe_docker.run("image", "command") == 151

    def test_container_remove_raises_returns_152(self):
        """
        Tests 152 is returned if an error is raised when calling container.remove
        """
        mock_container = create_mock_container()
        mock_container.remove.side_effect = RuntimeError("Something bad happened")
        with MockDockerClient(mock_container):
            safe_docker = SafeDockerClient()
            assert safe_docker.run("image", "command") == 152

    def test_main(self):
        """
        Tests main function against different command line arguments
        """
        tests = [
            # ( supplied command line arguments, expected call )
            (
                ["image", "command"],
                call("image", command="command", runtime=None, user=None, name=None, volumes={}, cap_add=[])
            ),
            (
                ["image", "command", "arg1", "arg2"],
                call("image", command="command arg1 arg2", runtime=None, user=None, name=None, volumes={}, cap_add=[])
            ),
            (
                ["--runtime", "nvidia", "image", "command"],
                call("image", command="command", runtime="nvidia", user=None, name=None, volumes={}, cap_add=[])
            ),
            (
                ["--user", "1001:1001", "image", "command"],
                call("image", command="command", runtime=None, user="1001:1001", name=None, volumes={}, cap_add=[])
            ),
            ([
                "--volume", "/local/path1:/container/path1",
                "--volume", "/local/path2:/container/path2:ro",
                "image",
                "command"
            ], call("image", command="command", runtime=None, user=None, name=None, volumes={
                "/local/path1": {
                    "bind": "/container/path1",
                    "mode": "rw"
                },
                "/local/path2": {
                    "bind": "/container/path2",
                    "mode": "ro"
                }
            }, cap_add=[])),
            ([
                "--runtime", "nvidia",
                "-u", "1001:1001",
                "-v", "/local/path1:/container/path1",
                "-v", "/local/path2:/container/path2:ro",
                "--cap-add", "bob",
                "--cap-add", "jimmy",
                "--name",
                "container_name",
                "image",
                "command",
                "arg1",
                "arg2"
            ], call(
                "image",
                command="command arg1 arg2",
                runtime="nvidia",
                user="1001:1001",
                name="container_name",
                volumes={
                    "/local/path1": {
                        "bind": "/container/path1",
                        "mode": "rw"
                    },
                    "/local/path2": {
                        "bind": "/container/path2",
                        "mode": "ro"
                    }
                }, cap_add=["bob", "jimmy"])
            )
        ]

        # Tests valid arguments
        mock_docker = create_autospec(SafeDockerClient)
        mock_docker.run.return_value = 0
        with patch("safe_docker_run.SafeDockerClient", return_value=mock_docker):
            for test in tests:
                arguments, expected_call = test
                main(arguments)
                assert mock_docker.run.call_args == expected_call

        # Tests invalid arguments
        tests = [
            [],
            None,
            ["image"],
            # Test some bad volume mounts
            ["-v", "bob", "image", "args"],
            ["-v", "/local/path", "image", "args"],
            ["-v", "/local/path:/container/path:blah", "image", "args"],
            ["-v", "", "image", "args"],
            ["-v", "a:b:c:d", "image", "args"]
        ]

        mock_docker = create_autospec(SafeDockerClient)
        with patch("safe_docker_run.SafeDockerClient", return_value=mock_docker):
            with self.assertRaises(SystemExit):
                for test in tests:
                    main(test)

    def test_clean_up(self):
        """
        Tests container clean up in case of SIGTERM and SIGINT
        """
        import subprocess
        import time
        import docker.errors

        docker_client = docker.from_env()
        container_name = "safedockertestcontainer1234"

        def get_container(name: str) -> Optional[Container]:
            try:
                return docker_client.containers.get(name)
            except docker.errors.NotFound:
                return None

        def remove_container_if_exists(name: str):
            container = get_container(name)
            if container:
                container.stop()
                container.remove()

        def wait_for_container(name: str) -> bool:
            for _ in itertools.count(5):
                if get_container(name):
                    return True
                time.sleep(1)
            return False

        # Clear any containers with container name
        remove_container_if_exists(container_name)

        # None => not signal is emitted - we should still finish with no containers at the end due
        # to the atexit
        for sig in [None, signal.SIGTERM, signal.SIGINT]:
            # Execute the safe docker run script in a different process
            proc = subprocess.Popen(['./safe_docker_run.py', "--name", container_name, "ubuntu:18.04", "sleep 10"])
            # NOTE: we need to wait for the container to come up as not all operating systems support blocking signals
            if wait_for_container(container_name) is False:
                raise RuntimeError("Test container did not come up")

            # Issue the signal and wait for the process to finish
            if sig:
                proc.send_signal(sig)
            proc.wait()

            # The container should no longer exist
            assert get_container(container_name) is None


if __name__ == '__main__':
    import nose
    nose.main()
