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
Docker login tests
"""
import os
import subprocess
import unittest
from unittest.mock import create_autospec, patch, call, MagicMock

import boto3
from boto3 import client
from botocore.stub import Stubber

from docker_login import login_dockerhub, logout_dockerhub, main, DOCKERHUB_RETRY_SECONDS, DOCKERHUB_LOGIN_NUM_RETRIES


SECRET_NAME = "secret_name"
SECRET_ENDPOINT_URL = "https://endpoint.url"
SECRET_ENDPOINT_REGION = "us-east-2"


def mock_boto(num_calls: int = 1):
    mock_client = client("secretsmanager", region_name="us-east-1")
    mock_session = create_autospec(boto3.Session)
    mock_session.client.return_value = mock_client

    # Stub get_secret_value response
    stub = Stubber(mock_client)
    for i in range(num_calls):
        stub.add_response(
            method="get_secret_value",
            expected_params={
                "SecretId": "secret_name"  # Matches os.environ['SECRET_NAME']
            }, service_response={
                "SecretString": """{"username": "myuser", "password": "mypass"}"""
            })
    return mock_session, stub


class TestDockerLogin(unittest.TestCase):

    @patch("subprocess.run", name="mock_subprocess_run")
    def test_docker_login_success(self, mock_run):
        """
        Tests successful docker login returns True and calls docker appropriately
        """
        mock_session, stub = mock_boto()
        stub.activate()
        with patch("boto3.Session", return_value=mock_session):
            mock_process = MagicMock(auto_spec=subprocess.Popen, name="mock_process")

            # Simulate successful login
            mock_process.returncode = 0
            mock_run.return_value = mock_process

            login_dockerhub(SECRET_NAME, SECRET_ENDPOINT_URL, SECRET_ENDPOINT_REGION)

            # Check boto client is properly created
            print(mock_session.client.call_args_list)
            assert mock_session.client.call_args_list == [
                call(service_name="secretsmanager", region_name="us-east-2", endpoint_url="https://endpoint.url")
            ]

            # Check that login call passes in the password in the correct way
            assert mock_run.call_args_list == [
                call(
                    ["docker", "login", "--username", "myuser", "--password-stdin"],
                    stdout=subprocess.PIPE,
                    input=str.encode("mypass")
                )
            ]
        stub.deactivate()

    @patch("subprocess.run", name="mock_subprocess_run")
    @patch("time.sleep")
    def test_docker_login_retry(self, mock_sleep, mock_run):
        """
        Tests retry mechanism
        """
        num_tries = 3
        mock_session, stub = mock_boto(num_calls=num_tries)
        stub.activate()
        with patch("boto3.Session", return_value=mock_session):
            mock_process = MagicMock(auto_spec=subprocess.Popen, name="mock_process")

            # Simulate successful login
            mock_process.returncode = 0

            # Simulate (num_tries - 1) errors + 1 success
            mock_run.side_effect = \
                [subprocess.CalledProcessError(1, "cmd", "some error")] * (num_tries - 1) + [mock_process]

            login_dockerhub(SECRET_NAME, SECRET_ENDPOINT_URL, SECRET_ENDPOINT_REGION)

            # Check boto client is properly created
            print(mock_session.client.call_args_list)
            assert mock_session.client.call_args_list == [
                call(service_name="secretsmanager", region_name="us-east-2", endpoint_url="https://endpoint.url")
            ] * num_tries

            # Check that login call passes in the password in the correct way
            cmd = ["docker", "login", "--username", "myuser", "--password-stdin"]
            assert mock_run.call_args_list == [
                call(cmd, stdout=subprocess.PIPE, input=str.encode("mypass"))
            ] * num_tries

            # Assert sleep was called appropriately
            assert mock_sleep.call_args_list == [
                call(2 ** retry_num * DOCKERHUB_RETRY_SECONDS) for retry_num in range(0, num_tries - 1)
            ]
        stub.deactivate()

    @patch("subprocess.run", name="mock_subprocess_run")
    @patch("time.sleep")
    def test_docker_login_retry_exhausted(self, mock_sleep, mock_run):
        """
        Tests retry mechanism
        """
        num_tries = DOCKERHUB_LOGIN_NUM_RETRIES
        mock_session, stub = mock_boto(num_calls=num_tries)
        stub.activate()
        with patch("boto3.Session", return_value=mock_session):
            # Simulate num_tries errors
            mock_run.side_effect = [subprocess.CalledProcessError(1, "cmd", "some error")] * num_tries

            with self.assertRaises(subprocess.CalledProcessError):
                login_dockerhub(SECRET_NAME, SECRET_ENDPOINT_URL, SECRET_ENDPOINT_REGION)

            # Check boto client is properly created
            assert mock_session.client.call_args_list == [
                call(service_name="secretsmanager", region_name="us-east-2", endpoint_url="https://endpoint.url")
            ] * num_tries

            # Check that login call passes in the password in the correct way
            cmd = ["docker", "login", "--username", "myuser", "--password-stdin"]
            assert mock_run.call_args_list == [
                call(cmd, stdout=subprocess.PIPE, input=str.encode("mypass"))
            ] * num_tries

            # Assert sleep was called appropriately
            assert mock_sleep.call_args_list == [
                call(2 ** retry_num * DOCKERHUB_RETRY_SECONDS) for retry_num in range(0, num_tries-1)
            ]
        stub.deactivate()

    @patch("subprocess.run", name="mock_subprocess_run")
    def test_docker_login_failed(self, mock_run):
        """
        Tests failed docker login return false
        """
        mock_session, stub = mock_boto()
        stub.activate()
        with patch("boto3.Session", return_value=mock_session):

            mock_process = MagicMock(auto_spec=subprocess.Popen, name="mock_process")

            # Simulate failed login
            mock_process.returncode = 1
            mock_run.return_value = mock_process

            with self.assertRaises(RuntimeError):
                login_dockerhub(SECRET_NAME, SECRET_ENDPOINT_URL, SECRET_ENDPOINT_REGION)
        stub.deactivate()

    @patch("subprocess.call", name="mock_subprocess_call")
    def test_logout(self, mock_call):
        """
        Tests logout calls docker command appropriately
        """
        logout_dockerhub()
        assert mock_call.call_args_list == [
            call(["docker", "logout"])
        ]

    @patch("docker_login.login_dockerhub")
    def test_main_exit(self, mock_login):
        """
        Tests main exits with error on failed docker login
        """
        mock_login.side_effect = RuntimeError("Didn't work")
        with self.assertRaises(SystemExit):
            main(["--secret-name", "name", "--secret-endpoint-url", "url", "--secret-endpoint-region", "r"])

    @patch("docker_login.login_dockerhub")
    def test_main_default_argument_values(self, mock_login):
        """
        Tests default arguments
        """

        # Good env
        env = {
            "DOCKERHUB_SECRET_ENDPOINT_URL": "url",
            "DOCKERHUB_SECRET_ENDPOINT_REGION": "region"
        }
        with patch.dict(os.environ, env):
            main(["--secret-name", "name"])
            assert mock_login.call_args_list == [
                call("name", "url", "region")
            ]

        # Bad envs - none or not all required vars defined
        tests = [
            {},
            {"DOCKERHUB_SECRET_ENDPOINT_URL": "url"},
            {"DOCKERHUB_SECRET_ENDPOINT_REGION": "region"}
        ]
        for bad_env in tests:
            with patch.dict(os.environ, bad_env):
                with self.assertRaises(RuntimeError):
                    main(["--secret-name", "name"])


if __name__ == '__main__':
    import nose
    nose.main()
