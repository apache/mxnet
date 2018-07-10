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

num_epoch = 2
model_name = 'lm_rnn_gluon_api'

context = mx.cpu(0)

def train(model, train_data):
    best_val = float("Inf")
    for epoch in range(args_epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx = context)
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, args_bptt)):
            data, target = get_batch(train_data, i)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and bptt size to balance it.
            gluon.utils.clip_global_norm(grads, args_clip * args_bptt * args_batch_size)

            trainer.step(args_batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % args_log_interval == 0 and ibatch > 0:
                cur_L = total_L / args_bptt / args_batch_size / args_log_interval
                print('[Epoch %d Batch %d] loss %.2f, perplexity %.2f' % (
                    epoch + 1, ibatch, cur_L, np.exp(cur_L)))
                total_L = 0.0

        val_L = eval(val_data, model)

        print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation perplexity %.2f' % (
            epoch + 1, time.time() - start_time, val_L, np.exp(val_L)))

        if val_L < best_val:
            best_val = val_L
            model.save_parameters(model_name + '.params')

def test(test_data, model):
    test_L = eval(test_data, model)
    return test_L, np.exp(test_L)

def save_inference_results(test, val):
    inference_results = dict()
    inference_results['val'] = val
    inference_results['test'] = test

    inference_results_file = model_name + '_inference' + '.json'

    # Write the inference results to local json file. This will be cleaned up later
    with open(inference_results_file, 'w') as file:
        json.dump(inference_results, file)

def clean_up_files (model_files):
    clean_ptb_data()
    clean_model_files(model_files)
    print ('Model files deleted')
    
def clean_model_files(model_files):
    for file in model_files:
        if os.path.isfile(file):
            os.remove(file)

if __name__=='__main__':
    ## If this code is being run on version >= 1.2.0 only then execute it, since it uses save_parameters and load_parameters API
    if compare_versions(str(mxnet_version), '1.2.1')  < 0:
        print ('Found MXNet version %s and exiting because this version does not contain save_parameters and load_parameters functions' %str(mxnet_version))
        sys.exit(1)
    
    corpus = Corpus(args_data)
    train_data = batchify(corpus.train, args_batch_size).as_in_context(context)
    val_data = batchify(corpus.valid, args_batch_size).as_in_context(context)
    test_data = batchify(corpus.test, args_batch_size).as_in_context(context)

    ntokens = len(corpus.dictionary)

    model = RNNModel(args_model, ntokens, args_emsize, args_nhid,
                       args_nlayers, args_dropout, args_tied)
    model.collect_params().initialize(mx.init.Xavier(), ctx=context)
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': args_lr, 'momentum': 0, 'wd': 0})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    train(model, train_data)
    val_loss, val_ppl = test(val_data, model)
    print('Validation loss %f, Validation perplexity %f'%(val_loss, val_ppl))
    test_loss, test_ppl = test(test_data, model)
    print('test loss %f, test perplexity %f'%(test_loss, test_ppl))

    val_results = dict()
    val_results['loss'] = val_loss
    val_results['ppl'] = val_ppl

    test_results = dict()
    test_results['loss'] = test_loss
    test_results['ppl'] = test_ppl

    save_inference_results(test_results, val_results)

    mxnet_folder = str(mxnet_version) + backslash + model_name + backslash

    files = list()
    files.append(model_name + '.params')
    files.append(model_name + '_inference' + '.json')
    upload_model_files_to_s3(bucket_name, files, mxnet_folder)
    clean_up_files(files)