"""Helper functions for parsing caffe prototxt into a workable DAG
"""


def process_network_proto(caffe_root, deploy_proto):
    """
    Runs the caffe upgrade tool on the prototxt to create a prototxt in the latest format.
    This enable us to work just with latest structures, instead of supporting all the variants

    :param caffe_root: link to caffe root folder, where the upgrade tool is located
    :param deploy_proto: name of the original prototxt file
    :return: name of new processed prototxt file
    """
    processed_deploy_proto = deploy_proto + ".processed"

    from shutil import copyfile
    copyfile(deploy_proto, processed_deploy_proto)

    # run upgrade tool on new file name (same output file)
    import os
    upgrade_tool_command_line = caffe_root + '/build/tools/upgrade_net_proto_text.bin ' \
                                + processed_deploy_proto + ' ' + processed_deploy_proto
    os.system(upgrade_tool_command_line)

    return processed_deploy_proto


class LayerRecord(object):

    def __init__(self, layer_def):

        self.layer_def = layer_def
        self.name = layer_def.name
        self.type = layer_def.type

        # keep filter, stride and pad
        if layer_def.type == 'Convolution':
            self.filter = list(layer_def.convolution_param.kernel_size)
            if len(self.filter) == 1:
                self.filter *= 2
            self.pad = list(layer_def.convolution_param.pad)
            if len(self.pad) == 0:
                self.pad = [0, 0]
            elif len(self.pad) == 1:
                self.pad *= 2
            self.stride = list(layer_def.convolution_param.stride)
            if len(self.stride) == 0:
                self.stride = [1, 1]
            elif len(self.stride) == 1:
                self.stride *= 2

        elif layer_def.type == 'Pooling':
            self.filter = [layer_def.pooling_param.kernel_size]
            if len(self.filter) == 1:
                self.filter *= 2
            self.pad = [layer_def.pooling_param.pad]
            if len(self.pad) == 0:
                self.pad = [0, 0]
            elif len(self.pad) == 1:
                self.pad *= 2
            self.stride = [layer_def.pooling_param.stride]
            if len(self.stride) == 0:
                self.stride = [1, 1]
            elif len(self.stride) == 1:
                self.stride *= 2

        else:
            self.filter = [0, 0]
            self.pad = [0, 0]
            self.stride = [1, 1]

        # keep tops
        self.tops = list(layer_def.top)

        # keep bottoms
        self.bottoms = list(layer_def.bottom)

        # list of parent layers
        self.parents = []

        # list of child layers
        self.children = []


def read_network_dag(processed_deploy_prototxt):
    """
    Reads from the caffe prototxt the network structure
    :param processed_deploy_prototxt: name of prototxt to load, preferably the prototxt should
     be processed before using a call to process_network_proto()
    :return: network_def, layer_name_to_record, top_to_layers
    network_def: caffe network structure, gives access to *all* the network information
    layer_name_to_record: *ordered* dictionary which maps between layer name and a structure which
      describes in a simple form the layer parameters
    top_to_layers: dictionary which maps a blob name to an ordered list of layers which output it
     when a top is used several times, like in inplace layhers, the list will contain all the layers
     by order of appearance
    """

    from caffe.proto import caffe_pb2
    from google.protobuf import text_format
    from collections import OrderedDict

    # load prototxt file
    network_def = caffe_pb2.NetParameter()
    with open(processed_deploy_prototxt, 'r') as proto_file:
        text_format.Merge(str(proto_file.read()), network_def)

    # map layer name to layer record
    layer_name_to_record = OrderedDict()
    for layer_def in network_def.layer:
        if (len(layer_def.include) == 0) or \
           (caffe_pb2.TEST in [item.phase for item in layer_def.include]):

            layer_name_to_record[layer_def.name] = LayerRecord(layer_def)

    top_to_layers = dict()
    for layer in network_def.layer:
        # no specific phase, or TEST phase is specifically asked for
        if (len(layer.include) == 0) or (caffe_pb2.TEST in [item.phase for item in layer.include]):
            for top in layer.top:
                if top not in top_to_layers:
                    top_to_layers[top] = list()
                top_to_layers[top].append(layer.name)

    # find parents and children of all layers
    for child_layer_name in layer_name_to_record.keys():
        child_layer_def = layer_name_to_record[child_layer_name]
        for bottom in child_layer_def.bottoms:
            for parent_layer_name in top_to_layers[bottom]:
                if parent_layer_name in layer_name_to_record:
                    parent_layer_def = layer_name_to_record[parent_layer_name]
                    if parent_layer_def not in child_layer_def.parents:
                        child_layer_def.parents.append(parent_layer_def)
                    if child_layer_def not in parent_layer_def.children:
                        parent_layer_def.children.append(child_layer_def)

    # update filter, strid, pad for maxout "structures"
    for layer_name in layer_name_to_record.keys():
        layer_def = layer_name_to_record[layer_name]
        if layer_def.type == 'Eltwise' and \
           len(layer_def.parents) == 1 and \
           layer_def.parents[0].type == 'Slice' and \
           len(layer_def.parents[0].parents) == 1 and \
           layer_def.parents[0].parents[0].type in ['Convolution', 'InnerProduct']:
            layer_def.filter = layer_def.parents[0].parents[0].filter
            layer_def.stride = layer_def.parents[0].parents[0].stride
            layer_def.pad = layer_def.parents[0].parents[0].pad

    return network_def, layer_name_to_record, top_to_layers


def read_caffe_mean(caffe_mean_file):
    """
    Reads caffe formatted mean file
    :param caffe_mean_file: path to caffe mean file, presumably with 'binaryproto' suffix
    :return: mean image, converted from BGR to RGB format
    """

    import caffe_parser
    import numpy as np
    mean_blob = caffe_parser.caffe_pb2.BlobProto()
    with open(caffe_mean_file, 'rb') as f:
        mean_blob.ParseFromString(f.read())

    img_mean_np = np.array(mean_blob.data)
    img_mean_np = img_mean_np.reshape(mean_blob.channels, mean_blob.height, mean_blob.width)

    # swap channels from Caffe BGR to RGB
    img_mean_np[[0, 2], :, :] = img_mean_np[[2, 0], :, :]

    return img_mean_np
