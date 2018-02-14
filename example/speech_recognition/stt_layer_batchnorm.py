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


def batchnorm(net,
              gamma=None,
              beta=None,
              eps=0.001,
              momentum=0.9,
              fix_gamma=False,
              use_global_stats=False,
              output_mean_var=False,
              name=None):
    if gamma is not None and beta is not None:
        net = mx.sym.BatchNorm(data=net,
                               gamma=gamma,
                               beta=beta,
                               eps=eps,
                               momentum=momentum,
                               fix_gamma=fix_gamma,
                               use_global_stats=use_global_stats,
                               output_mean_var=output_mean_var,
                               name=name
                               )
    else:
        net = mx.sym.BatchNorm(data=net,
                               eps=eps,
                               momentum=momentum,
                               fix_gamma=fix_gamma,
                               use_global_stats=use_global_stats,
                               output_mean_var=output_mean_var,
                               name=name
                               )
    return net
