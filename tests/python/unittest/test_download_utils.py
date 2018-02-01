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

import mxnet as mx
import mock
import requests
import requests_mock
from nose.tools import *

DATASET_URL = 'http://fakeyannlecun.com/fake.dataset.tar.gz'
DATASET_FILENAME = 'fake.dataset.tar.gz'


@raises(requests.exceptions.HTTPError)
def test_fake_yann_http_error():
    # Patch requests so we don't actually issue an http request.
    with requests_mock.mock() as request_mock:
        # Patch time so we don't waste time retrying.
        with mock.patch('time.sleep'):
            request_mock.get(DATASET_URL, text='Not Found', status_code=404)
            mx.utils.download(DATASET_URL, overwrite=True)


def test_fake_yann_success():
    # Patch requests so we don't actually issue an http request.
    with requests_mock.mock() as m:
        m.get(DATASET_URL, text='data')
        result = mx.utils.download(DATASET_URL, overwrite=True)
        assert result == DATASET_FILENAME


if __name__ == '__main__':

    import nose
    nose.runmodule()
