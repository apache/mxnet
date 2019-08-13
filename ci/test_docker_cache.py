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
Distributed Docker cache tests
"""

import unittest.mock
import tempfile
import os
import logging
import subprocess
import sys
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(__file__))
import docker_cache
import build as build_util

DOCKERFILE_DIR = 'docker'
DOCKER_REGISTRY_NAME = 'test_registry'
DOCKER_REGISTRY_PORT = 5000
DOCKER_REGISTRY_PATH = 'localhost:{}'.format(DOCKER_REGISTRY_PORT)

class RedirectSubprocessOutput(object):
    """
    Redirect the output of all subprocess.call calls to a readable buffer instead of writing it to stdout/stderr.
    The output can then be retrieved with get_output.
    """
    def __enter__(self):
        self.buf_output = tempfile.TemporaryFile()

        def trampoline(*popenargs, **kwargs):
            self.call(*popenargs, **kwargs)

        self.old_method = subprocess.call
        subprocess.call = trampoline
        return self

    def __exit__(self, *args):
        logging.info('Releasing docker output buffer:\n%s', self.get_output())
        subprocess.call = self.old_method
        self.buf_output.close()

    def call(self, *popenargs, **kwargs):
        """
        Replace subprocess.call
        :param popenargs:
        :param timeout:
        :param kwargs:
        :return:
        """
        kwargs['stderr'] = subprocess.STDOUT
        kwargs['stdout'] = self.buf_output
        return self.old_method(*popenargs, **kwargs)

    def get_output(self):
        self.buf_output.seek(0)
        return self.buf_output.read().decode('utf-8')


class TestDockerCache(unittest.TestCase):
    """
    Test utility class
    """
    def setUp(self):
        logging.getLogger().setLevel(logging.DEBUG)

        # We need to be in the same directory than the script so the commands in the dockerfiles work as
        # expected. But the script can be invoked from a different path
        base = os.path.split(os.path.realpath(__file__))[0]
        os.chdir(base)

        docker_cache._login_dockerhub = MagicMock()  # Override login

        # Stop in case previous execution was dirty
        try:
            self._stop_local_docker_registry()
        except Exception:
            pass

        # Start up docker registry
        self._start_local_docker_registry()

    def tearDown(self):
        # Stop docker registry
        self._stop_local_docker_registry()

    @classmethod
    def _start_local_docker_registry(cls):
        # https://docs.docker.com/registry/deploying/#run-a-local-registrys
        start_cmd = [
            'docker', 'run', '-d', '-p', '{}:{}'.format(DOCKER_REGISTRY_PORT, DOCKER_REGISTRY_PORT),
            '--name', DOCKER_REGISTRY_NAME, 'registry:2'
        ]
        subprocess.check_call(start_cmd)

    @classmethod
    def _stop_local_docker_registry(cls):
        # https://docs.docker.com/registry/deploying/#run-a-local-registry
        stop_cmd = ['docker', 'container', 'stop', DOCKER_REGISTRY_NAME]
        subprocess.check_call(stop_cmd)

        clean_cmd = ['docker', 'container', 'rm', '-v', DOCKER_REGISTRY_NAME]
        subprocess.check_call(clean_cmd)

    def test_full_cache(self):
        """
        Test whether it's possible to restore cache entirely
        :return:
        """
        dockerfile_content = """
                FROM busybox
                RUN touch ~/file1
                RUN touch ~/file2
                RUN touch ~/file3
                RUN touch ~/file4
                """
        platform = 'test_full_cache'
        docker_tag = build_util.get_docker_tag(platform=platform, registry=DOCKER_REGISTRY_PATH)
        dockerfile_path = os.path.join(DOCKERFILE_DIR, 'Dockerfile.' + platform)
        try:
            with open(dockerfile_path, 'w') as dockerfile_handle:
                dockerfile_handle.write(dockerfile_content)

            # Warm up
            docker_cache.delete_local_docker_cache(docker_tag=docker_tag)

            def warm_up_lambda_func():
                build_util.build_docker(docker_binary='docker', platform=platform, registry=DOCKER_REGISTRY_PATH)
            _assert_docker_build(lambda_func=warm_up_lambda_func, expected_cache_hit_count=0,
                                 expected_cache_miss_count=4)

            # Assert local cache is properly primed
            def primed_cache_lambda_func():
                build_util.build_docker(docker_binary='docker', platform=platform, registry=DOCKER_REGISTRY_PATH)
            _assert_docker_build(lambda_func=primed_cache_lambda_func, expected_cache_hit_count=4,
                                 expected_cache_miss_count=0)

            # Upload and clean local cache
            docker_cache.build_save_containers(platforms=[platform], registry=DOCKER_REGISTRY_PATH, load_cache=False)
            docker_cache.delete_local_docker_cache(docker_tag=docker_tag)

            # Build with clean local cache and cache loading enabled
            def clean_cache_lambda_func():
                docker_cache.build_save_containers(
                    platforms=[platform], registry=DOCKER_REGISTRY_PATH, load_cache=True)
            _assert_docker_build(lambda_func=clean_cache_lambda_func, expected_cache_hit_count=4,
                                 expected_cache_miss_count=0)
        finally:
            # Delete dockerfile
            os.remove(dockerfile_path)
            docker_cache.delete_local_docker_cache(docker_tag=docker_tag)



    def test_partial_cache(self):
        """
        Test whether it's possible to restore cache and then pit it up partially by using a Dockerfile which shares
        some parts
        :return:
        """
        # These two dockerfiles diverge at the fourth RUN statement. Their common parts (1-3) should be re-used
        dockerfile_content_1 = """
                FROM busybox
                RUN touch ~/file1
                RUN touch ~/file2
                RUN touch ~/file3
                RUN touch ~/file4
                """
        dockerfile_content_2 = """
                FROM busybox
                RUN touch ~/file1
                RUN touch ~/file2
                RUN touch ~/file3
                RUN touch ~/file5
                RUN touch ~/file4
                RUN touch ~/file6
                """
        platform = 'test_partial_cache'
        docker_tag = build_util.get_docker_tag(platform=platform, registry=DOCKER_REGISTRY_PATH)
        dockerfile_path = os.path.join(DOCKERFILE_DIR, 'Dockerfile.' + platform)
        try:
            # Write initial Dockerfile
            with open(dockerfile_path, 'w') as dockerfile_handle:
                dockerfile_handle.write(dockerfile_content_1)

            # Warm up
            docker_cache.delete_local_docker_cache(docker_tag=docker_tag)

            def warm_up_lambda_func():
                build_util.build_docker(docker_binary='docker', platform=platform, registry=DOCKER_REGISTRY_PATH)
            _assert_docker_build(lambda_func=warm_up_lambda_func, expected_cache_hit_count=0,
                                 expected_cache_miss_count=4)

            # Assert local cache is properly primed
            def primed_cache_lambda_func():
                build_util.build_docker(docker_binary='docker', platform=platform, registry=DOCKER_REGISTRY_PATH)
            _assert_docker_build(lambda_func=primed_cache_lambda_func, expected_cache_hit_count=4,
                                 expected_cache_miss_count=0)

            # Upload and clean local cache
            docker_cache.build_save_containers(platforms=[platform], registry=DOCKER_REGISTRY_PATH, load_cache=False)
            docker_cache.delete_local_docker_cache(docker_tag=docker_tag)

            # Replace Dockerfile with the second one, resulting in a partial cache hit
            with open(dockerfile_path, 'w') as dockerfile_handle:
                dockerfile_handle.write(dockerfile_content_2)

            # Test if partial cache is properly hit. It will attempt to load the cache from the first Dockerfile,
            # resulting in a partial hit
            def partial_cache_lambda_func():
                docker_cache.build_save_containers(
                    platforms=[platform], registry=DOCKER_REGISTRY_PATH, load_cache=True)
            _assert_docker_build(lambda_func=partial_cache_lambda_func, expected_cache_hit_count=3,
                                 expected_cache_miss_count=3)

        finally:
            # Delete dockerfile
            os.remove(dockerfile_path)
            docker_cache.delete_local_docker_cache(docker_tag=docker_tag)


def _assert_docker_build(lambda_func, expected_cache_hit_count: int, expected_cache_miss_count: int):
    with RedirectSubprocessOutput() as redirected_output:
        lambda_func()
        output = redirected_output.get_output()
        assert output.count('Running in') == expected_cache_miss_count, \
            'Expected {} "Running in", got {}. Log:{}'.\
                format(expected_cache_miss_count, output.count('Running in'), output)
        assert output.count('Using cache') == expected_cache_hit_count, \
            'Expected {} "Using cache", got {}. Log:{}'.\
                format(expected_cache_hit_count, output.count('Using cache'), output)


if __name__ == '__main__':
    import nose
    nose.main()
