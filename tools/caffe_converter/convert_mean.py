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

"""Convert caffe mean
"""
import argparse
import numpy as np
import mxnet as mx
import caffe_parser

def convert_mean(binaryproto_fname, output=None):
    """Convert caffe mean

    Parameters
    ----------
    binaryproto_fname : str
        Filename of the mean
    output : str, optional
        Save the mean into mxnet's format

    Returns
    -------
    NDArray
        Mean in ndarray
    """
    mean_blob = caffe_parser.caffe_pb2.BlobProto()
    with open(binaryproto_fname, 'rb') as f:
        mean_blob.ParseFromString(f.read())

    img_mean_np = np.array(mean_blob.data)
    img_mean_np = img_mean_np.reshape(
        mean_blob.channels, mean_blob.height, mean_blob.width
    )
    # swap channels from Caffe BGR to RGB
    img_mean_np[[0, 2], :, :] = img_mean_np[[2, 0], :, :]
    nd = mx.nd.array(img_mean_np)
    if output is not None:
        mx.nd.save(output, {"mean_image": nd})
    return nd

def main():
    parser = argparse.ArgumentParser(description='Convert caffe mean')
    parser.add_argument('binaryproto_fname', help='Filename of the mean')
    parser.add_argument('output', help='The name of the output file')
    args = parser.parse_args()
    convert_mean(args.binaryproto_fname, args.output)

if __name__ == '__main__':
    main()
