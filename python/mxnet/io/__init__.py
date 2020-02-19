#!/usr/bin/env python

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

# coding: utf-8
# pylint: disable=wildcard-import
""" Data iterators for common data formats and utility functions."""

from . import io
from .io import CSVIter, DataBatch, DataDesc, DataIter, ImageDetRecordIter, ImageRecordInt8Iter, ImageRecordIter,\
    ImageRecordIter_v1, ImageRecordUInt8Iter, ImageRecordUInt8Iter_v1, LibSVMIter, MNISTIter, MXDataIter, NDArrayIter,\
    PrefetchingIter, ResizeIter

from . import utils
from .utils import _init_data, _getdata_by_idx, _has_instance
