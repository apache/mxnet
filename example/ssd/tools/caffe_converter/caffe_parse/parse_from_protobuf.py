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

from google.protobuf import text_format
import numpy as np
import caffe_parse.caffe_pb2 as caffe_pb2


def parse_caffemodel(file_path):
    """
    parses the trained .caffemodel file

    filepath: /path/to/trained-model.caffemodel

    returns: layers
    """
    f = open(file_path, 'rb')
    contents = f.read()

    net_param = caffe_pb2.NetParameter()
    net_param.ParseFromString(contents)

    layers = find_layers(net_param)
    return layers


def find_layers(net_param):
    if len(net_param.layers) > 0:
        return net_param.layers
    elif len(net_param.layer) > 0:
        return net_param.layer
    else:
        raise Exception("Couldn't find layers")


def main():
    param_dict = parse_caffemodel('xxx.caffemodel')


if __name__ == '__main__':
    main()
