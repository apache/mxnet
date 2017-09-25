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

from __future__ import print_function
import mxnet as mx
x = mx.th.randn(2, 2, ctx=mx.cpu(0))
print(x.asnumpy())
y = mx.th.abs(x)
print(y.asnumpy())

x = mx.th.randn(2, 2, ctx=mx.cpu(0))
print(x.asnumpy())
mx.th.abs(x, x) # in-place
print(x.asnumpy())

x = mx.th.ones(2, 2, ctx=mx.cpu(0))
y = mx.th.ones(2, 2, ctx=mx.cpu(0))*2
print(mx.th.cdiv(x,y).asnumpy())
