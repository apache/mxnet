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
import tempfile
import warnings

try:
    from unittest import mock
except ImportError:
    import mock
import mxnet as mx
import requests
from nose.tools import raises


class MockResponse(requests.Response):
    def __init__(self, status_code, content):
        super(MockResponse, self).__init__()
        assert isinstance(status_code, int)
        self.status_code = status_code
        self.raw = io.BytesIO(content.encode('utf-8'))


@raises(Exception)
@mock.patch(
    'requests.get', mock.Mock(side_effect=requests.exceptions.ConnectionError))
def test_download_retries():
    mx.gluon.utils.download("http://doesnotexist.notfound")


@mock.patch(
    'requests.get',
    mock.Mock(side_effect=
              lambda *args, **kwargs: MockResponse(200, 'MOCK CONTENT' * 100)))
def test_download_successful():
    tmp = tempfile.mkdtemp()
    tmpfile = os.path.join(tmp, 'README.md')
    mx.gluon.utils.download(
        "https://raw.githubusercontent.com/apache/incubator-mxnet/master/README.md",
        path=tmpfile)
    assert os.path.getsize(tmpfile) > 100


@mock.patch(
    'requests.get',
    mock.Mock(
        side_effect=lambda *args, **kwargs: MockResponse(200, 'MOCK CONTENT')))
def test_download_ssl_verify():
    with warnings.catch_warnings(record=True) as warnings_:
        mx.gluon.utils.download(
            "https://mxnet.incubator.apache.org/index.html", verify_ssl=False)
    assert any(
        str(w.message).startswith('Unverified HTTPS request')
        for w in warnings_)
