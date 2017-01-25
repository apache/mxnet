import mxnet as mx
import numpy as np
import argparse

caffe_flag = True
try:
    import caffe
    from caffe.proto import caffe_pb2
except ImportError:
    caffe_flag = False
    import caffe_parse.caffe_pb2


def protoBlobFileToND(proto_file):
    data = ''
    file = open(proto_file, "r")
    if not file:
        raise Exception("ERROR (" + proto_file + ")!")
    data = file.read()
    file.close()

    if caffe_flag:
        mean_blob = caffe.proto.caffe_pb2.BlobProto()
    else:
        mean_blob = caffe_parse.caffe_pb2.BlobProto()

    mean_blob.ParseFromString(data)
    img_mean_np = np.array(mean_blob.data)
    img_mean_np = img_mean_np.reshape(
        mean_blob.channels, mean_blob.height, mean_blob.width
    )
    # swap channels from Caffe BGR to RGB
    img_mean_np2 = img_mean_np
    img_mean_np[0] = img_mean_np2[2]
    img_mean_np[2] = img_mean_np2[0]
    return mx.nd.array(img_mean_np)


def main():
    parser = argparse.ArgumentParser(description='Caffe prototxt to mxnet model parameter converter.\
                    Note that only basic functions are implemented. You are welcomed to contribute to this file.')
    parser.add_argument('mean_image_proto', help='The protobuf file in Caffe format')
    parser.add_argument('save_name', help='The name of the output file prefix')
    args = parser.parse_args()
    nd = protoBlobFileToND(args.mean_image_proto)
    mx.nd.save(args.save_name + ".nd", {"mean_image": nd})


if __name__ == '__main__':
    main()
