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

from .onnx2mx.import_model import import_model, get_model_metadata
from .onnx2mx.import_to_gluon import import_to_gluon
from ...onnx.export_model import export_model
print('Calling ONNX module through mxnet.contrib.onnx...')
print('Please be advised that mxnet.contrib.onnx.import_model/get_model_metadata/import_to_gluon '
      'will be deprecated in the upcoming MXNet v1.10 release. mxnet.contrib.onnx.export_model '
      'has been moved to mxnet.onnx.export_model. The mxnet.contrib.onnx.export_model '
      'alias will also be deprecated in the MXNet v1.10 release.')
