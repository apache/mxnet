"""Parse caffe's protobuf
"""
import re
try:
    import caffe
    from caffe.proto import caffe_pb2
    use_caffe = True
except ImportError:
    try:
        import caffe_pb2
    except ImportError:
        raise ImportError('You used to compile with protoc --python_out=./ ./caffe.proto')
    use_caffe = False

from google.protobuf import text_format

def read_prototxt(fname):
    """Return a caffe_pb2.NetParameter object that defined in a prototxt file
    """
    proto = caffe_pb2.NetParameter()
    with open(fname, 'r') as f:
        text_format.Merge(str(f.read()), proto)
    return proto

def get_layers(proto):
    """Returns layers in a caffe_pb2.NetParameter object
    """
    if len(proto.layer):
        return proto.layer
    elif len(proto.layers):
        return proto.layers
    else:
        raise ValueError('Invalid proto file.')

def read_caffemodel(prototxt_fname, caffemodel_fname):
    """Return a caffe_pb2.NetParameter object that defined in a binary
    caffemodel file
    """
    if use_caffe:
        caffe.set_mode_cpu()
        net = caffe.Net(prototxt_fname, caffemodel_fname, caffe.TEST)
        layer_names = net._layer_names
        layers = net.layers
        return (layers, layer_names)
    else:
        proto = caffe_pb2.NetParameter()
        with open(caffemodel_fname, 'rb') as f:
            proto.ParseFromString(f.read())
        return (get_layers(proto), None)

def layer_iter(layers, layer_names):
    if use_caffe:
        for layer_idx, layer in enumerate(layers):
            layer_name = re.sub('[-/]', '_', layer_names[layer_idx])
            layer_type = layer.type
            layer_blobs = layer.blobs
            yield (layer_name, layer_type, layer_blobs)
    else:
        for layer in layers:
            layer_name = re.sub('[-/]', '_', layer.name)
            layer_type = layer.type
            layer_blobs = layer.blobs
            yield (layer_name, layer_type, layer_blobs)
