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


def test_module_checkpoint_api():
    model_name = 'module_checkpoint_api'
    print ('Performing inference for model/API %s' % model_name)

    # For each MXNet version that has the saved models
    for folder in get_top_level_folders_in_bucket(s3, model_bucket_name):
        logging.info('Fetching files for MXNet version : %s and model %s' % (folder, model_name))
        model_files = download_model_files_from_s3(model_name, folder)
        if len(model_files) == 0:
            logging.warn('No training files found for %s for MXNet version : %s' % (model_name, folder))
            continue

        data = mx.nd.load(''.join([model_name, '-data']))
        data_iter = mx.io.NDArrayIter(data['data'], data['labels'], batch_size=10)
        # Load the model and perform inference
        loaded_model = get_module_api_model_definition()

        sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, 1)
        loaded_model.bind(data_shapes=data_iter.provide_data, label_shapes=data_iter.provide_label)
        loaded_model.set_params(arg_params, aux_params)

        old_inference_results = load_inference_results(model_name)
        inference_results = loaded_model.predict(data_iter)
        # Check whether they are equal or not ?
        assert_almost_equal(inference_results.asnumpy(), old_inference_results.asnumpy(), rtol=rtol_default, atol=atol_default)
        clean_model_files(model_files, model_name)
        logging.info('=================================')

    logging.info('Assertion passed for model : %s' % model_name)


def test_lenet_gluon_load_params_api():
    model_name = 'lenet_gluon_save_params_api'
    logging.info('Performing inference for model/API %s' % model_name)

    for folder in get_top_level_folders_in_bucket(s3, model_bucket_name):
        logging.info('Fetching files for MXNet version : %s and model %s' % (folder, model_name))
        model_files = download_model_files_from_s3(model_name, folder)
        if len(model_files) == 0:
            logging.warn('No training files found for %s for MXNet version : %s' % (model_name, folder))
            continue

        data = mx.nd.load(''.join([model_name, '-data']))
        test_data = data['data']
        # Load the model and perform inference
        loaded_model = Net()
        loaded_model.load_params(model_name + '-params')
        output = loaded_model(test_data)
        old_inference_results = mx.nd.load(model_name + '-inference')['inference']
        assert_almost_equal(old_inference_results.asnumpy(), output.asnumpy(), rtol=rtol_default, atol=atol_default)
        clean_model_files(model_files, model_name)
        logging.info('=================================')
    logging.info('Assertion passed for model : %s' % model_name)


def test_lenet_gluon_hybrid_imports_api():
    model_name = 'lenet_gluon_hybrid_export_api'
    logging.info('Performing inference for model/API %s' % model_name)

    for folder in get_top_level_folders_in_bucket(s3, model_bucket_name):
        logging.info('Fetching files for MXNet version : %s and model %s' % (folder, model_name))
        model_files = download_model_files_from_s3(model_name, folder)
        if len(model_files) == 0:
            logging.warn('No training files found for %s for MXNet version : %s' % (model_name, folder))
            continue
            # Load the model and perform inference
        data = mx.nd.load(''.join([model_name, '-data']))
        test_data = data['data']
        loaded_model = HybridNet()
        loaded_model = gluon.SymbolBlock.imports(model_name + '-symbol.json', ['data'], model_name + '-0000.params')
        output = loaded_model(test_data)
        old_inference_results = mx.nd.load(model_name + '-inference')['inference']
        assert_almost_equal(old_inference_results.asnumpy(), output.asnumpy(), rtol=rtol_default, atol=atol_default)
        clean_model_files(model_files, model_name)
        logging.info('=================================')
    logging.info('Assertion passed for model : %s' % model_name)


def test_lstm_gluon_load_parameters_api():
    # If this code is being run on version >= 1.2.0 only then execute it,
    # since it uses save_parameters and load_parameters API

    if compare_versions(str(mxnet_version), '1.2.1') < 0:
        logging.warn('Found MXNet version %s and exiting because this version does not contain save_parameters'
                     ' and load_parameters functions' % str(mxnet_version))
        return

    model_name = 'lstm_gluon_save_parameters_api'
    logging.info('Performing inference for model/API %s and model' % model_name)

    for folder in get_top_level_folders_in_bucket(s3, model_bucket_name):
        logging.info('Fetching files for MXNet version : %s' % folder)
        model_files = download_model_files_from_s3(model_name, folder)
        if len(model_files) == 0:
            logging.warn('No training files found for %s for MXNet version : %s' % (model_name, folder))
            continue

        data = mx.nd.load(''.join([model_name, '-data']))
        test_data = data['data']
        # Load the model and perform inference
        loaded_model = SimpleLSTMModel()
        loaded_model.load_parameters(model_name + '-params')
        output = loaded_model(test_data)
        old_inference_results = mx.nd.load(model_name + '-inference')['inference']
        assert_almost_equal(old_inference_results.asnumpy(), output.asnumpy(), rtol=rtol_default, atol=atol_default)
        clean_model_files(model_files, model_name)
        logging.info('=================================')
    logging.info('Assertion passed for model : %s' % model_name)


if __name__ == '__main__':
    test_module_checkpoint_api()
    test_lenet_gluon_load_params_api()
    test_lenet_gluon_hybrid_imports_api()
    test_lstm_gluon_load_parameters_api()
