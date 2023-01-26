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

import io
import os
import warnings
import glob
import shutil
import multiprocessing as mp

try:
    from unittest import mock
except ImportError:
    import mock
import mxnet as mx
import requests
import pytest


class MockResponse(requests.Response):
    def __init__(self, status_code, content):
        super(MockResponse, self).__init__()
        assert isinstance(status_code, int)
        self.status_code = status_code
        self.raw = io.BytesIO(content.encode('utf-8'))


@mock.patch(
    'requests.get', mock.Mock(side_effect=requests.exceptions.ConnectionError))
def test_download_retries_error():
    with pytest.raises(Exception):
        mx.gluon.utils.download("http://doesnotexist.notfound")


@mock.patch(
    'requests.get',
    mock.Mock(side_effect=lambda *args, **kwargs: MockResponse(200, 'MOCK CONTENT' * 100)))
def _download_successful(tmp):
    """ internal use for testing download successfully """
    mx.gluon.utils.download(
        "https://raw.githubusercontent.com/apache/mxnet/master/README.md",
        path=tmp)


def test_download_successful(tmpdir):
    """ test download with one process """
    tmp = str(tmpdir)
    tmpfile = os.path.join(tmp, 'README.md')
    _download_successful(tmpfile)
    assert os.path.getsize(tmpfile) > 100, os.path.getsize(tmpfile)
    pattern = os.path.join(tmp, 'README.md*')
    # check only one file we want left
    assert len(glob.glob(pattern)) == 1, glob.glob(pattern)
    # delete temp dir
    shutil.rmtree(tmp)


def test_multiprocessing_download_successful(tmpdir):
    """ test download with multiprocessing """
    tmp = str(tmpdir)
    tmpfile = os.path.join(tmp, 'README.md')
    process_list = []
    # test it with 10 processes
    for i in range(10):
        process_list.append(mp.Process(
            target=_download_successful, args=(tmpfile,)))
        process_list[i].start()
    for i in range(10):
        process_list[i].join()
    assert os.path.getsize(tmpfile) > 100, os.path.getsize(tmpfile)
    # check only one file we want left
    pattern = os.path.join(tmp, 'README.md*')
    assert len(glob.glob(pattern)) == 1, glob.glob(pattern)
    # delete temp dir
    shutil.rmtree(tmp)


@mock.patch(
    'requests.get',
    mock.Mock(
        side_effect=lambda *args, **kwargs: MockResponse(200, 'MOCK CONTENT')))
def test_download_ssl_verify():
    """ test download verify_ssl parameter """
    with warnings.catch_warnings(record=True) as warnings_:
        mx.gluon.utils.download(
            "https://mxnet.apache.org/index.html", verify_ssl=False)
    assert any(
        str(w.message).startswith('Unverified HTTPS request')
        for w in warnings_)
