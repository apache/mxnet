"""Test converted models
"""
import os
import argparse
import sys
import logging
import mxnet as mx
from convert_caffe_modelzoo import convert_caffe_model, get_model_meta_info, download_caffe_model
from compare_layers import convert_and_compare_caffe_to_mxnet

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../../example/image-classification"))
from test_score import download_data  # pylint: disable=wrong-import-position
from score import score # pylint: disable=wrong-import-position
logging.basicConfig(level=logging.DEBUG)

def test_imagenet_model_performance(model_name, val_data, gpus, batch_size):
    """test model performance on imagenet """
    logging.info('test performance of model: %s', model_name)
    meta_info = get_model_meta_info(model_name)
    [model_name, mean] = convert_caffe_model(model_name, meta_info)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, 0)
    acc = [mx.metric.create('acc'), mx.metric.create('top_k_accuracy', top_k=5)]
    if isinstance(mean, str):
        mean_args = {'mean_img':mean}
    else:
        mean_args = {'rgb_mean':','.join([str(i) for i in mean])}

    print(val_data)
    gpus_string = '' if gpus[0] == -1 else ','.join([str(i) for i in gpus])
    (speed,) = score(model=(sym, arg_params, aux_params),
                     data_val=val_data,
                     label_name='prob_label',
                     metrics=acc,
                     gpus=gpus_string,
                     batch_size=batch_size,
                     max_num_examples=500,
                     **mean_args)
    logging.info('speed : %f image/sec', speed)
    for a in acc:
        logging.info(a.get())
    max_performance_diff_allowed = 0.03
    assert acc[0].get()[1] > meta_info['top-1-acc'] - max_performance_diff_allowed
    assert acc[1].get()[1] > meta_info['top-5-acc'] - max_performance_diff_allowed


def test_model_weights_and_outputs(model_name, image_url, gpu):
    """
    Run the layer comparison on one of the known caffe models.
    :param model_name: available models are listed in convert_caffe_modelzoo.py
    :param image_url: image file or url to run inference on
    :param gpu: gpu to use, -1 for cpu
    """

    logging.info('test weights and outputs of model: %s', model_name)
    meta_info = get_model_meta_info(model_name)

    (prototxt, caffemodel, mean) = download_caffe_model(model_name, meta_info, dst_dir='./model')
    convert_and_compare_caffe_to_mxnet(image_url, gpu, prototxt, caffemodel, mean,
                                       mean_diff_allowed=1e-03, max_diff_allowed=1e-01)

    return


def main():
    """Entrypoint for test_converter"""
    parser = argparse.ArgumentParser(description='Test Caffe converter')
    parser.add_argument('--cpu', action='store_true', help='use cpu?')
    parser.add_argument('--image_url', type=str,
                        default='http://writm.com/wp-content/uploads/2016/08/Cat-hd-wallpapers.jpg',
                        help='input image to test inference, can be either file path or url')
    args = parser.parse_args()
    if args.cpu:
        gpus = [-1]
        batch_size = 32
    else:
        gpus = mx.test_utils.list_gpus()
        assert gpus, 'At least one GPU is needed to run test_converter in GPU mode'
        batch_size = 32 * len(gpus)

    models = ['bvlc_googlenet', 'vgg-16', 'resnet-50']

    val = download_data()
    for m in models:
        test_model_weights_and_outputs(m, args.image_url, gpus[0])
        test_imagenet_model_performance(m, val, gpus, batch_size)

if __name__ == '__main__':
    main()
