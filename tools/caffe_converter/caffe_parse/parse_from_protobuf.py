from google.protobuf import text_format
import numpy as np
import caffe_pb2

def parse_caffemodel(filepath):
    '''
    parses the trained .caffemodel file

    filepath: /path/to/trained-model.caffemodel

    returns: layers
    '''
    f = open(filepath, 'rb')
    contents = f.read()

    netparam = caffe_pb2.NetParameter()
    netparam.ParseFromString(contents)

    layers = find_layers(netparam)
    return layers

def find_layers(netparam):
    if len(netparam.layers) > 0:
        return netparam.layers
    elif len(netparam.layer) > 0:
        return netparam.layer
    else:
        raise Exception ("Couldn't find layers")

def main():
    param_dict = parse_caffemodel('xxx.caffemodel')

if __name__ == '__main__':
    main()
