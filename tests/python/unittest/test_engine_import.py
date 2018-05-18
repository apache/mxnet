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

import os
import sys

def test_engine_import():
    import mxnet
    def test_import():
        version = sys.version_info
        if version >= (3, 4):
            import importlib
            importlib.reload(mxnet)
        elif version >= (3, ):
            import imp
            imp.reload(mxnet)
        else:
            reload(mxnet)
    engine_types = ['', 'NaiveEngine', 'ThreadedEngine', 'ThreadedEnginePerDevice']

    for type in engine_types:
        if not type:
            os.environ.pop('MXNET_ENGINE_TYPE', None)
        else:
            os.environ['MXNET_ENGINE_TYPE'] = type
        test_import()

if __name__ == '__main__':
    import nose
    nose.runmodule()
