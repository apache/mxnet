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

"""Test converted models layer by layer
"""
import os
import argparse
import logging
import mxnet as mx
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)


def read_image(img_path, image_dims=None, mean=None):
    """
    Reads an image from file path or URL, optionally resizing to given image dimensions and
    subtracting mean.
    :param img_path: path to file, or url to download
    :param image_dims: image dimensions to resize to, or None
    :param mean: mean file to subtract, or None
    :return: loaded image, in RGB format
    """

    import urllib

    filename = img_path.split("/")[-1]
    if img_path.startswith('http'):
        urllib.urlretrieve(img_path, filename)
        img = cv2.imread(filename)
    else:
        img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if image_dims is not None:
        img = cv2.resize(img, image_dims)  # resize to image_dims to fit model
    img = np.rollaxis(img, 2) # change to (c, h, w) order
    img = img[np.newaxis, :]  # extend to (n, c, h, w)
    if mean is not None:
        mean = np.array(mean)
        if mean.shape == (3,):
            mean = mean[np.newaxis, :, np.newaxis, np.newaxis]  # extend to (n, c, 1, 1)
        img = img.astype(np.float32) - mean # subtract mean

    return img


def _ch_dev(arg_params, aux_params, ctx):
    """
    Changes device of given mxnet arguments
    :param arg_params: arguments
    :param aux_params: auxiliary parameters
    :param ctx: new device context
    :return: arguments and auxiliary parameters on new device
    """
    new_args = dict()
    new_auxs = dict()
    for k, v in arg_params.items():
        new_args[k] = v.as_in_context(ctx)
    for k, v in aux_params.items():
        new_auxs[k] = v.as_in_context(ctx)
    return new_args, new_auxs


def convert_and_compare_caffe_to_mxnet(image_url, gpu, caffe_prototxt_path, caffe_model_path,
                                       caffe_mean, mean_diff_allowed, max_diff_allowed):
    """
    Run the layer comparison on a caffe model, given its prototxt, weights and mean.
    The comparison is done by inferring on a given image using both caffe and mxnet model
    :param image_url: image file or url to run inference on
    :param gpu: gpu to use, -1 for cpu
    :param caffe_prototxt_path: path to caffe prototxt
    :param caffe_model_path: path to caffe weights
    :param caffe_mean: path to caffe mean file
    """

    import caffe
    from caffe_proto_utils import read_network_dag, process_network_proto, read_caffe_mean
    from convert_model import convert_model

    if isinstance(caffe_mean, str):
        caffe_mean = read_caffe_mean(caffe_mean)
    elif caffe_mean is None:
        pass
    elif len(caffe_mean) == 3:
        # swap channels from Caffe BGR to RGB
        caffe_mean = caffe_mean[::-1]

    # get caffe root location, this is needed to run the upgrade network utility, so we only need
    # to support parsing of latest caffe
    caffe_root = os.path.dirname(os.path.dirname(caffe.__path__[0]))
    caffe_prototxt_path = process_network_proto(caffe_root, caffe_prototxt_path)

    _, layer_name_to_record, top_to_layers = read_network_dag(caffe_prototxt_path)

    caffe.set_mode_cpu()
    caffe_net = caffe.Net(caffe_prototxt_path, caffe_model_path, caffe.TEST)

    image_dims = tuple(caffe_net.blobs['data'].shape)[2:4]

    logging.info('getting image %s', image_url)
    img_rgb = read_image(image_url, image_dims, caffe_mean)
    img_bgr = img_rgb[:, ::-1, :, :]

    caffe_net.blobs['data'].reshape(*img_bgr.shape)
    caffe_net.blobs['data'].data[...] = img_bgr
    _ = caffe_net.forward()

    # read sym and add all outputs
    sym, arg_params, aux_params, _ = convert_model(caffe_prototxt_path, caffe_model_path)
    sym = sym.get_internals()

    # now mxnet
    if gpu < 0:
        ctx = mx.cpu(0)
    else:
        ctx = mx.gpu(gpu)

    arg_params, aux_params = _ch_dev(arg_params, aux_params, ctx)
    arg_params["data"] = mx.nd.array(img_rgb, ctx)
    arg_params["prob_label"] = mx.nd.empty((1,), ctx)
    exe = sym.bind(ctx, arg_params, args_grad=None, grad_req="null", aux_states=aux_params)
    exe.forward(is_train=False)

    compare_layers_from_nets(caffe_net, arg_params, aux_params, exe, layer_name_to_record,
                             top_to_layers, mean_diff_allowed, max_diff_allowed)

    return


def _bfs(root_node, process_node):
    """
    Implementation of Breadth-first search (BFS) on caffe network DAG
    :param root_node: root node of caffe network DAG
    :param process_node: function to run on each node
    """

    from collections import deque

    seen_nodes = set()
    next_nodes = deque()

    seen_nodes.add(root_node)
    next_nodes.append(root_node)

    while next_nodes:
        current_node = next_nodes.popleft()

        # process current node
        process_node(current_node)

        for child_node in current_node.children:
            if child_node not in seen_nodes:
                seen_nodes.add(child_node)
                next_nodes.append(child_node)


def compare_layers_from_nets(caffe_net, arg_params, aux_params, exe, layer_name_to_record,
                             top_to_layers, mean_diff_allowed, max_diff_allowed):
    """
    Compare layer by layer of a caffe network with mxnet network
    :param caffe_net: loaded caffe network
    :param arg_params: arguments
    :param aux_params: auxiliary parameters
    :param exe: mxnet model
    :param layer_name_to_record: map between caffe layer and information record
    :param top_to_layers: map between caffe blob name to layers which outputs it (including inplace)
    :param mean_diff_allowed: mean difference allowed between caffe blob and mxnet blob
    :param max_diff_allowed: max difference allowed between caffe blob and mxnet blob
    """

    import re

    log_format = '  {0:<40}  {1:<40}  {2:<8}  {3:>10}  {4:>10}  {5:<1}'

    compare_layers_from_nets.is_first_convolution = True

    def _compare_blob(caf_blob, mx_blob, caf_name, mx_name, blob_type, note):
        diff = np.abs(mx_blob - caf_blob)
        diff_mean = diff.mean()
        diff_max = diff.max()
        logging.info(log_format.format(caf_name, mx_name, blob_type, '%4.5f' % diff_mean,
                                       '%4.5f' % diff_max, note))
        assert diff_mean < mean_diff_allowed
        assert diff_max < max_diff_allowed

    def _process_layer_parameters(layer):

        logging.debug('processing layer %s of type %s', layer.name, layer.type)

        normalized_layer_name = re.sub('[-/]', '_', layer.name)

        # handle weight and bias of convolution and fully-connected layers
        if layer.name in caffe_net.params and layer.type in ['Convolution', 'InnerProduct',
                                                             'Deconvolution']:

            has_bias = len(caffe_net.params[layer.name]) > 1

            mx_name_weight = '{}_weight'.format(normalized_layer_name)
            mx_beta = arg_params[mx_name_weight].asnumpy()

            # first convolution should change from BGR to RGB
            if layer.type == 'Convolution' and compare_layers_from_nets.is_first_convolution:
                compare_layers_from_nets.is_first_convolution = False

                # if RGB or RGBA
                if mx_beta.shape[1] == 3 or mx_beta.shape[1] == 4:
                    # Swapping BGR of caffe into RGB in mxnet
                    mx_beta[:, [0, 2], :, :] = mx_beta[:, [2, 0], :, :]

            caf_beta = caffe_net.params[layer.name][0].data
            _compare_blob(caf_beta, mx_beta, layer.name, mx_name_weight, 'weight', '')

            if has_bias:
                mx_name_bias = '{}_bias'.format(normalized_layer_name)
                mx_gamma = arg_params[mx_name_bias].asnumpy()
                caf_gamma = caffe_net.params[layer.name][1].data
                _compare_blob(caf_gamma, mx_gamma, layer.name, mx_name_bias, 'bias', '')

        elif layer.name in caffe_net.params and layer.type == 'Scale':

            if 'scale' in normalized_layer_name:
                bn_name = normalized_layer_name.replace('scale', 'bn')
            elif 'sc' in normalized_layer_name:
                bn_name = normalized_layer_name.replace('sc', 'bn')
            else:
                assert False, 'Unknown name convention for bn/scale'

            beta_name = '{}_beta'.format(bn_name)
            gamma_name = '{}_gamma'.format(bn_name)

            mx_beta = arg_params[beta_name].asnumpy()
            caf_beta = caffe_net.params[layer.name][1].data
            _compare_blob(caf_beta, mx_beta, layer.name, beta_name, 'mov_mean', '')

            mx_gamma = arg_params[gamma_name].asnumpy()
            caf_gamma = caffe_net.params[layer.name][0].data
            _compare_blob(caf_gamma, mx_gamma, layer.name, gamma_name, 'mov_var', '')

        elif layer.name in caffe_net.params and layer.type == 'BatchNorm':

            mean_name = '{}_moving_mean'.format(normalized_layer_name)
            var_name = '{}_moving_var'.format(normalized_layer_name)

            caf_rescale_factor = caffe_net.params[layer.name][2].data

            mx_mean = aux_params[mean_name].asnumpy()
            caf_mean = caffe_net.params[layer.name][0].data / caf_rescale_factor
            _compare_blob(caf_mean, mx_mean, layer.name, mean_name, 'mean', '')

            mx_var = aux_params[var_name].asnumpy()
            caf_var = caffe_net.params[layer.name][1].data / caf_rescale_factor
            _compare_blob(caf_var, mx_var, layer.name, var_name, 'var',
                          'expect 1e-04 change due to cudnn eps')

        elif layer.type in ['Input', 'Pooling', 'ReLU', 'Eltwise', 'Softmax', 'LRN', 'Concat',
                            'Dropout', 'Crop']:
            # no parameters to check for these layers
            pass

        else:
            logging.warn('No handling for layer %s of type %s, should we ignore it?', layer.name,
                         layer.type)

        return

    def _process_layer_output(caffe_blob_name):

        logging.debug('processing blob %s', caffe_blob_name)

        # skip blobs not originating from actual layers, e.g. artificial split layers added by caffe
        if caffe_blob_name not in top_to_layers:
            return

        caf_blob = caffe_net.blobs[caffe_blob_name].data

        # data should change from BGR to RGB
        if caffe_blob_name == 'data':

            # if RGB or RGBA
            if caf_blob.shape[1] == 3 or caf_blob.shape[1] == 4:
                # Swapping BGR of caffe into RGB in mxnet
                caf_blob[:, [0, 2], :, :] = caf_blob[:, [2, 0], :, :]
            mx_name = 'data'

        else:
            # get last layer name which outputs this blob name
            last_layer_name = top_to_layers[caffe_blob_name][-1]
            normalized_last_layer_name = re.sub('[-/]', '_', last_layer_name)
            mx_name = '{}_output'.format(normalized_last_layer_name)
            if 'scale' in mx_name:
                mx_name = mx_name.replace('scale', 'bn')
            elif 'sc' in mx_name:
                mx_name = mx_name.replace('sc', 'bn')

        if mx_name not in exe.output_dict:
            logging.error('mxnet blob %s is missing, time to extend the compare tool..', mx_name)
            return

        mx_blob = exe.output_dict[mx_name].asnumpy()
        _compare_blob(caf_blob, mx_blob, caffe_blob_name, mx_name, 'output', '')

        return

    # check layer parameters
    logging.info('\n***** Network Parameters '.ljust(140, '*'))
    logging.info(log_format.format('CAFFE', 'MXNET', 'Type', 'Mean(diff)', 'Max(diff)', 'Note'))
    first_layer_name = layer_name_to_record.keys()[0]
    _bfs(layer_name_to_record[first_layer_name], _process_layer_parameters)

    # check layer output
    logging.info('\n***** Network Outputs '.ljust(140, '*'))
    logging.info(log_format.format('CAFFE', 'MXNET', 'Type', 'Mean(diff)', 'Max(diff)', 'Note'))
    for caffe_blob_name in caffe_net.blobs.keys():
        _process_layer_output(caffe_blob_name)

    return


def main():
    """Entrypoint for compare_layers"""

    parser = argparse.ArgumentParser(
        description='Tool for testing caffe to mxnet conversion layer by layer')
    parser.add_argument('--image_url', type=str,
                        default='http://writm.com/wp-content/uploads/2016/08/Cat-hd-wallpapers.jpg',
                        help='input image to test inference, can be either file path or url')
    parser.add_argument('--caffe_prototxt_path', type=str,
                        default='./model.prototxt',
                        help='path to caffe prototxt')
    parser.add_argument('--caffe_model_path', type=str,
                        default='./model.caffemodel',
                        help='path to caffe weights')
    parser.add_argument('--caffe_mean', type=str,
                        default='./model_mean.binaryproto',
                        help='path to caffe mean file')
    parser.add_argument('--mean_diff_allowed', type=int, default=1e-03,
                        help='mean difference allowed between caffe blob and mxnet blob')
    parser.add_argument('--max_diff_allowed', type=int, default=1e-01,
                        help='max difference allowed between caffe blob and mxnet blob')
    parser.add_argument('--gpu', type=int, default=-1, help='the gpu id used for predict')
    args = parser.parse_args()
    convert_and_compare_caffe_to_mxnet(args.image_url, args.gpu, args.caffe_prototxt_path,
                                       args.caffe_model_path, args.caffe_mean,
                                       args.mean_diff_allowed, args.max_diff_allowed)

if __name__ == '__main__':
    main()
