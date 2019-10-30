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
# 
import mxnet as mx
from core.model import get_model

def test_model():
    def test_ncf(model_type):
        net = get_model(model_type=model_type, factor_size_mlp=128, factor_size_gmf=64, 
                        model_layers=[256, 128, 64], num_hidden=1, max_user=138493, max_item=26744)
        mod = mx.module.Module(net, context=mx.cpu(), data_names=['user', 'item'], label_names=['softmax_label'])
        provide_data = [mx.io.DataDesc(name='item', shape=((1,))),
                        mx.io.DataDesc(name='user', shape=((1,)))]
        provide_label = [mx.io.DataDesc(name='softmax_label', shape=((1,)))]
        mod.bind(for_training=True, data_shapes=provide_data, label_shapes=provide_label)
        mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
        data = [mx.nd.full(shape=shape, val=26744, ctx=mx.cpu(), dtype='int32')
                for _, shape in mod.data_shapes]
        batch = mx.io.DataBatch(data, [])
        mod.forward(batch)
        mod.backward()
        mx.nd.waitall()

        data_dict = {'user': data[0], 'item': data[1]}
        calib_data = mx.io.NDArrayIter(data=data_dict, batch_size=1)
        calib_data = mx.test_utils.DummyIter(calib_data)
        arg_params, aux_params = mod.get_params()
        qsym, qarg_params, qaux_params = mx.contrib.quant.quantize_model_mkldnn(sym=net,
                                                                                arg_params=arg_params,
                                                                                aux_params=aux_params,
                                                                                ctx=mx.cpu(),
                                                                                quantized_dtype='auto',
                                                                                calib_mode='naive',
                                                                                calib_data=calib_data,
                                                                                data_names=['user', 'item'],
                                                                                excluded_sym_names=['post_gemm_concat', 'fc_final'],
                                                                                num_calib_examples=1)
        qmod = mx.module.Module(qsym, context=mx.cpu(), data_names=['user', 'item'], label_names=['softmax_label'])
        qmod.bind(for_training=True, data_shapes=provide_data, label_shapes=provide_label)
        qmod.set_params(qarg_params, qaux_params)
        qmod.forward(batch)
        mx.nd.waitall()

    for model_type in ['neumf', 'mlp', 'gmf']:
        test_ncf(model_type)

if __name__ == "__main__":
    import nose
    nose.runmodule()

