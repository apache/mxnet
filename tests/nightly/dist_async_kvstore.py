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

# pylint: skip-file
import sys
sys.path.insert(0, "../../python/")
import mxnet as mx

kv = mx.kv.create('dist_async')
my_rank = kv.rank
nworker = kv.num_workers

def test_gluon_trainer_type():
    def check_trainer_kv_update(weight_stype, update_on_kv):
        params = mx.gluon.ParameterDict()
        x = params.get('x', shape=(10,1), lr_mult=1.0, stype=weight_stype)
        params.initialize(ctx=[mx.cpu(0), mx.cpu(1)], init='zeros')
        try:
            trainer = mx.gluon.Trainer(params, 'sgd', {'learning_rate': 0.1},
                                       kvstore=kv, update_on_kvstore=update_on_kv)
            trainer._init_kvstore()
            assert trainer._kv_initialized
            assert trainer._update_on_kvstore is True
        except ValueError:
            assert update_on_kv is False

    check_trainer_kv_update('default', False)
    check_trainer_kv_update('default', True)
    check_trainer_kv_update('default', None)
    check_trainer_kv_update('row_sparse', False)
    check_trainer_kv_update('row_sparse', True)
    check_trainer_kv_update('row_sparse', None)
    print('worker ' + str(my_rank) + ' passed test_gluon_trainer_type')

if __name__ == "__main__":
    test_gluon_trainer_type()
