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


class STTBucketingModule(mx.mod.BucketingModule):

    def save_checkpoint(self, prefix, epoch, save_optimizer_states=False):
        symbol, data_names, label_names = self._sym_gen(self._default_bucket_key)
        symbol.save('%s-symbol.json' % prefix)
        param_name = '%s-%04d.params' % (prefix, epoch)
        self.save_params(param_name)
        if save_optimizer_states:
            state_name = '%s-%04d.states' % (prefix, epoch)
            self._curr_module.save_optimizer_states(state_name)
