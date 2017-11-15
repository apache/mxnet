from __future__ import print_function
import argparse
import mxnet as mx
import numpy as np

import caffe
from caffe.proto import caffe_pb2

class CaffeModelConverter:
    def __init__(self):
        self.dict_param = {}
    
    def add_param(self, param_name, layer_index, blob_index):
        blobs = self.layers[layer_index].blobs
        self.dict_param[param_name] = mx.nd.array(caffe.io.blobproto_to_array(blobs[blob_index]))
    
    def add_arg_param(self, param_name, layer_index, blob_index):
        self.add_param('arg:%s' % param_name, layer_index, blob_index)

    def add_aux_param(self, param_name, layer_index, blob_index):
        self.add_param('aux:%s' % param_name, layer_index, blob_index)

    def add_optional_arg_param(self, param_name, layer_index, blob_index):
        blobs = self.layers[layer_index].blobs
        if(blob_index < len(blobs)):
            self.add_arg_param(param_name, layer_index, blob_index)

    def convert(self, caffemodel_path, outmodel_path):
        self.net_param = caffe_pb2.NetParameter()
        with open(caffemodel_path, 'rb') as f:
            self.net_param.ParseFromString(f.read())
        
        layers = self.net_param.layer
        self.layers = layers
        
        for idx, layer in enumerate(layers):
            layer_name = str(layer.name)

            if len(layer.blobs) > 0:

                layer_type = layer.type

                # If this is a layer that has only weight and bias as parameter
                if layer_type == 'Convolution' or layer_type == 'InnerProduct' or layer_type == 'Deconvolution':

                    # Add weight and bias to the dictionary
                    self.add_arg_param('%s_weight' % layer_name, layer_index=idx, blob_index=0)
                    self.add_optional_arg_param('%s_bias' % layer_name, layer_index=idx, blob_index=1)

                elif layer_type == 'BatchNorm':

                    gamma_param_name = '%s_gamma' % layer_name
                    beta_param_name = '%s_beta' % layer_name

                    next_layer = layers[idx + 1]

                    if next_layer.type == 'Scale':
                        # If next layer is scale layer, get gamma and beta from there
                        self.add_arg_param(gamma_param_name, layer_index=idx+1, blob_index=0)
                        self.add_arg_param(beta_param_name, layer_index=idx+1, blob_index=1)
                    
                    mean_param_name = '%s_moving_mean' % layer_name
                    var_param_name = '%s_moving_var' % layer_name
                    
                    self.add_aux_param(mean_param_name, layer_index=idx, blob_index=0)
                    self.add_aux_param(var_param_name, layer_index=idx, blob_index=1)
                    
                elif layer_type == 'Scale':
                
                    prev_layer = layers[idx - 1]
                    
                    if prev_layer.type == 'BatchNorm':
                        continue
                    else:
                        # Use the naming convention used by CaffeOp
                        self.add_arg_param('%s_0_weight' % layer_name, layer_index=idx, blob_index=0)
                        self.add_optional_arg_param('%s_1_bias' % layer_name, layer_index=idx, blob_index=1)
        
        mx.nd.save(outmodel_path, self.dict_param)

def main():
    parser = argparse.ArgumentParser(description='.caffemodel to MXNet .params converter.')
    parser.add_argument('caffemodel', help='Path to the .caffemodel file to convert.')
    parser.add_argument('output_file_name', help='Name of the output .params file.')

    args = parser.parse_args()

    converter = CaffeModelConverter()
    converter.convert(args.caffemodel, args.output_file_name)

if __name__ == '__main__':
    main()

