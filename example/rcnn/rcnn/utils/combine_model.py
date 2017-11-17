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

from load_model import load_checkpoint
from save_model import save_checkpoint


def combine_model(prefix1, epoch1, prefix2, epoch2, prefix_out, epoch_out):
    args1, auxs1 = load_checkpoint(prefix1, epoch1)
    args2, auxs2 = load_checkpoint(prefix2, epoch2)
    arg_names = args1.keys() + args2.keys()
    aux_names = auxs1.keys() + auxs2.keys()
    args = dict()
    for arg in arg_names:
        if arg in args1:
            args[arg] = args1[arg]
        else:
            args[arg] = args2[arg]
    auxs = dict()
    for aux in aux_names:
        if aux in auxs1:
            auxs[aux] = auxs1[aux]
        else:
            auxs[aux] = auxs2[aux]
    save_checkpoint(prefix_out, epoch_out, args, auxs)
