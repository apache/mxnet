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

from common import *


def train_module_checkpoint_api():
    model_name = 'module_checkpoint_api'
    create_model_folder(model_name)
    logging.info('Saving files for model %s' % model_name)
    # Prepare data
    test_data = mx.nd.array(np.random.uniform(-1, 1, size=(20, 1)))
    test_label = mx.nd.array(np.random.randint(0, 2, size=(20,)), dtype='float32')
    data_iter = mx.io.NDArrayIter(test_data, test_label, batch_size=10)

    mod = get_module_api_model_definition()
    mod.bind(data_shapes=data_iter.provide_data, label_shapes=data_iter.provide_label)
    weights = mx.initializer.Xavier(magnitude=2.57)
    mod.init_params(weights)

    mod.save_checkpoint(os.path.join(get_model_path(model_name), model_name), 1)

    inference_results = mod.predict(data_iter)
    # Save inference_results
    # Save the model files
    save_data_and_labels(test_data, test_label, model_name)
    save_inference_results(inference_results, model_name)


def train_lenet_gluon_save_params_api():
    model_name = 'lenet_gluon_save_params_api'
    create_model_folder(model_name)
    logging.info('Saving files for model %s' % model_name)
    net = Net()
    weights = mx.initializer.Xavier(magnitude=2.57)
    net.initialize(weights, ctx=[mx.cpu(0)])
    # Prepare data

    test_data = mx.nd.array(np.random.uniform(-1, 1, size=(20, 1, 30, 30)))
    output = net(test_data)
    # print (y)

    mx.nd.save(os.path.join(get_model_path(model_name), ''.join([model_name, '-data'])), {'data': test_data})
    save_inference_results(output, model_name)
    net.save_params(os.path.join(get_model_path(model_name), ''.join([model_name, '-params'])))


def train_lenet_gluon_hybrid_export_api():
    model_name = 'lenet_gluon_hybrid_export_api'
    logging.info('Saving files for model %s' % model_name)
    create_model_folder(model_name)
    net = HybridNet()
    weights = mx.initializer.Xavier(magnitude=2.57)
    net.initialize(weights, ctx=[mx.cpu(0)])
    net.hybridize()
    # Prepare data
    test_data = mx.nd.array(np.random.uniform(-1, 1, size=(20, 1, 30, 30)))
    output = net(test_data)
    # print (y)
    # Save the test data as well.
    # Save the inference output ys
    # Save the model params

    mx.nd.save(os.path.join(get_model_path(model_name), ''.join([model_name, '-data'])), {'data': test_data})
    save_inference_results(output, model_name)
    if compare_versions(str(mxnet_version) , '1.1.0') < 0:
        # v1.0.0 does not have the epoch param in the .exports API. Hence adding this safety net
        net.export(os.path.join(get_model_path(model_name), model_name))
    else:
        # Saving with 0 since by default on 1.0.0 it was saved with 0, so simplifying things
        net.export(os.path.join(get_model_path(model_name), model_name), epoch=0)



def train_lstm_gluon_save_parameters_api():
    # If this code is being run on version >= 1.2.1 only then execute it,
    # since it uses save_parameters and load_parameters API
    if compare_versions(str(mxnet_version), '1.2.1') < 0:
        logging.warn('Found MXNet version %s and exiting because this version does not contain save_parameters'
                     ' and load_parameters functions' % str(mxnet_version))
        return

    model_name = 'lstm_gluon_save_parameters_api'
    logging.info('Saving files for model %s' % model_name)
    create_model_folder(model_name)
    net = SimpleLSTMModel()
    weights = mx.initializer.Xavier(magnitude=2.57)
    net.initialize(weights, ctx=[mx.cpu(0)])

    test_data = mx.nd.array(np.random.uniform(-1, 1, size=(10, 30)))
    output = net(test_data)
    # print output
    mx.nd.save(os.path.join(get_model_path(model_name), ''.join([model_name, '-data'])), {'data': test_data})
    save_inference_results(output, model_name)
    net.save_parameters(os.path.join(get_model_path(model_name), ''.join([model_name, '-params'])))


def create_root_folder():
    base_path = os.getcwd()
    version_path = os.path.join(base_path, 'models')
    if not os.path.exists(version_path):
        os.mkdir(version_path)


if __name__ == '__main__':
    create_root_folder()

    train_module_checkpoint_api()
    train_lenet_gluon_save_params_api()
    train_lenet_gluon_hybrid_export_api()
    train_lstm_gluon_save_parameters_api()
