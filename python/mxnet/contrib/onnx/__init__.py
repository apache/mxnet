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
"""Module for ONNX model format support for Apache MXNet."""

from ...onnx import export_model as export_model_

def export_model(*args, **kwargs):
    print('Calling mxnet.contrib.onnx.export_model...')
    print('Please be advised that the ONNX module has been moved to mxnet.onnx and '
          'mxnet.onnx.export_model is the preferred path. The current path will be deprecated '
          'in the upcoming MXNet v1.10 release.')
    return export_model_(*args, **kwargs)
