from __future__ import print_function
import argparse
import mxnet as mx
import numpy as np

import caffe
from caffe.proto import caffe_pb2

class CaffeModelConverter:
    def __init__(self):
        self.dict_param = {}
    
    def add_arg_param(self, param_name, blob_index):
        self.dict_param['arg:%s' % param_name] = mx.nd.array(caffe.io.blobproto_to_array(self.blobs[blob_index]))

    def add_optional_arg_param(self, param_name, blob_index):
        if(blob_index < len(self.blobs)):
            self.add_arg_param(param_name, blob_index)

    def convert(self, caffemodel_path, outmodel_path):
        self.net_param = caffe_pb2.NetParameter()
        with open(caffemodel_path, 'rb') as f:
            self.net_param.ParseFromString(f.read())
        
        for layer in self.net_param.layer:
            layer_name = str(layer.name)

            self.blobs = layer.blobs

            if len(self.blobs) > 0:
                layer_type = layer.type
                if layer_type == 'Convolution' or layer_type == 'InnerProduct':
                    self.add_arg_param('%s_weight' % layer_name, blob_index=0)
                    self.add_optional_arg_param('%s_bias' % layer_name, blob_index=1)
        
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

