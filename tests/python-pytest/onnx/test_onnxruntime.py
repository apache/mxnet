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

import mxnet as mx
import numpy as np
import gluoncv
import onnxruntime

from mxnet.test_utils import assert_almost_equal
from common import with_seed

import json
import os
import pytest
import shutil



class GluonModel():
    def __init__(self, model_name, input_shape, input_dtype, tmpdir):
        self.model_name = model_name
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.modelpath = os.path.join(tmpdir, model_name)
        self.ctx = mx.cpu(0)
        self.get_model()
        self.export()

    def get_model(self):
        self.model = gluoncv.model_zoo.get_model(self.model_name, pretrained=True, ctx=self.ctx)
        self.model.hybridize()

    def export(self):
        data = mx.nd.zeros(self.input_shape, dtype=self.input_dtype, ctx=self.ctx)
        self.model.forward(data)
        self.model.export(self.modelpath, 0)

    def export_onnx(self):
        onnx_file = self.modelpath + ".onnx"
        mx.contrib.onnx.export_model(self.modelpath + "-symbol.json", self.modelpath + "-0000.params",
                                     [self.input_shape], self.input_dtype, onnx_file)
        return onnx_file

    def predict(self, data):
        return self.model(data)


def download_test_images(image_urls, tmpdir):
    from urllib.parse import urlparse
    paths = []
    for url in image_urls:
        filename = os.path.join(tmpdir, os.path.basename(urlparse(url).path))
        mx.test_utils.download(url, fname=filename)
        paths.append(filename)
    return paths


@with_seed()
@pytest.mark.parametrize('model', ['bert_12_768_12'])
def test_bert_inference_onnxruntime(tmp_path, model):
    tmp_path = str(tmp_path)
    try:
        import gluonnlp as nlp
        dataset = 'book_corpus_wiki_en_uncased'
        ctx = mx.cpu(0)
        model, vocab = nlp.model.get_model(
            name=model,
            ctx=ctx,
            dataset_name=dataset,
            pretrained=False,
            use_pooler=True,
            use_decoder=False,
            use_classifier=False)
        model.initialize(ctx=ctx)
        model.hybridize(static_alloc=True)

        batch = 5
        seq_length = 16
        # create synthetic test data
        inputs = mx.nd.random.uniform(0, 30522, shape=(batch, seq_length), dtype='float32')
        token_types = mx.nd.random.uniform(0, 2, shape=(batch, seq_length), dtype='float32')
        valid_length = mx.nd.array([seq_length] * batch, dtype='float32')

        seq_encoding, cls_encoding = model(inputs, token_types, valid_length)

        prefix = "%s/bert" % tmp_path
        model.export(prefix)
        sym_file = "%s-symbol.json" % prefix
        params_file = "%s-0000.params" % prefix
        onnx_file = "%s.onnx" % prefix


        input_shapes = [(batch, seq_length), (batch, seq_length), (batch,)]
        converted_model_path = mx.contrib.onnx.export_model(sym_file, params_file, input_shapes, np.float32, onnx_file)


        # create onnxruntime session using the generated onnx file
        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)
        onnx_inputs = [inputs, token_types, valid_length]
        input_dict = dict((session.get_inputs()[i].name, onnx_inputs[i].asnumpy()) for i in range(len(onnx_inputs)))
        pred_onx, cls_onx = session.run(None, input_dict)

        assert_almost_equal(seq_encoding, pred_onx, rtol=0.01, atol=0.01)
        assert_almost_equal(cls_encoding, cls_onx, rtol=0.01, atol=0.01)

    finally:
        shutil.rmtree(tmp_path)


@pytest.mark.parametrize('model', [
    'alexnet',
    'cifar_resnet20_v1',
    'cifar_resnet56_v1',
    'cifar_resnet110_v1',
    'cifar_resnet20_v2',
    'cifar_resnet56_v2',
    'cifar_resnet110_v2',
    'cifar_wideresnet16_10',
    'cifar_wideresnet28_10',
    'cifar_wideresnet40_8',
    'cifar_resnext29_16x64d',
    'darknet53',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    'googlenet',
    'mobilenet1.0',
    'mobilenet0.75',
    'mobilenet0.5',
    'mobilenet0.25',
    'mobilenetv2_1.0',
    'mobilenetv2_0.75',
    'mobilenetv2_0.5',
    'mobilenetv2_0.25',
    'mobilenetv3_large',
    'mobilenetv3_small',
    'resnest14',
    'resnest26',
    'resnest50',
    'resnest101',
    'resnest200',
    'resnest269',
    'resnet18_v1',
    'resnet18_v1b_0.89',
    'resnet18_v2',
    'resnet34_v1',
    'resnet34_v2',
    'resnet50_v1',
    'resnet50_v1d_0.86',
    'resnet50_v1d_0.48',
    'resnet50_v1d_0.37',
    'resnet50_v1d_0.11',
    'resnet50_v2',
    'resnet101_v1',
    'resnet101_v1d_0.76',
    'resnet101_v1d_0.73',
    'resnet101_v2',
    'resnet152_v1',
    'resnet152_v2',
    'resnext50_32x4d',
    'resnext101_32x4d',
    'resnext101_64x4d',
    'senet_154',
    'se_resnext101_32x4d',
    'se_resnext101_64x4d',
    'se_resnext50_32x4d',
    'squeezenet1.0',
    'squeezenet1.1',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19',
    'vgg19_bn',
    'xception',
    'inceptionv3'
])
def test_obj_class_model_inference_onnxruntime(tmp_path, model):
    inlen = 299 if 'inceptionv3' == model else 224
    def normalize_image(imgfile):
        img_data = mx.image.imread(imgfile)
        img_data = mx.image.imresize(img_data, inlen, inlen)
        img_data = img_data.transpose([2, 0, 1]).astype('float32')
        mean_vec = mx.nd.array([0.485, 0.456, 0.406])
        stddev_vec = mx.nd.array([0.229, 0.224, 0.225])
        norm_img_data = mx.nd.zeros(img_data.shape).astype('float32')
        for i in range(img_data.shape[0]):
            norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        return norm_img_data.reshape(1, 3, inlen, inlen).astype('float32')

    try:
        tmp_path = str(tmp_path)
        M = GluonModel(model, (1,3,inlen,inlen), 'float32', tmp_path)
        onnx_file = M.export_onnx()

        # create onnxruntime session using the generated onnx file
        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)
        input_name = session.get_inputs()[0].name

        test_image_urls = [
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/bikers.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/car.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/dancer.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/duck.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/fieldhockey.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/flower.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/runners.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/shark.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/soccer2.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/tree.jpg',
        ]

        for img in download_test_images(test_image_urls, tmp_path):
            img_data = normalize_image(img)
            mx_result = M.predict(img_data)
            onnx_result = session.run([], {input_name: img_data.asnumpy()})[0]
            assert_almost_equal(mx_result, onnx_result)

    finally:
        shutil.rmtree(tmp_path)




@pytest.mark.parametrize('model', [
    'center_net_resnet18_v1b_voc',
    'center_net_resnet50_v1b_voc',
    'center_net_resnet101_v1b_voc',
    'center_net_resnet18_v1b_coco',
    'center_net_resnet50_v1b_coco',
    'center_net_resnet101_v1b_coco',
    # the following models are failing due to onnxruntime errors
    #'ssd_300_vgg16_atrous_voc',
    #'ssd_512_vgg16_atrous_voc',
    #'ssd_512_resnet50_v1_voc',
    #'ssd_512_mobilenet1.0_voc',
    #'faster_rcnn_resnet50_v1b_voc',
    #'yolo3_darknet53_voc',
    #'yolo3_mobilenet1.0_voc',
    #'ssd_300_vgg16_atrous_coco',
    #'ssd_512_vgg16_atrous_coco',
    #'ssd_300_resnet34_v1b_coco',
    #'ssd_512_resnet50_v1_coco',
    #'ssd_512_mobilenet1.0_coco',
    #'faster_rcnn_resnet50_v1b_coco',
    #'faster_rcnn_resnet101_v1d_coco',
    #'yolo3_darknet53_coco',
    #'yolo3_mobilenet1.0_coco',
])
def test_obj_detection_model_inference_onnxruntime(tmp_path, model):
    def normalize_image(imgfile):
        img = mx.image.imread(imgfile)
        img, _ = mx.image.center_crop(img, size=(512, 512))
        img, _ = gluoncv.data.transforms.presets.center_net.transform_test(img, short=512)
        return img

    try:
        tmp_path = str(tmp_path)
        M = GluonModel(model, (1,3,512,512), 'float32', tmp_path)
        onnx_file = M.export_onnx()
        # create onnxruntime session using the generated onnx file
        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)
        input_name = session.get_inputs()[0].name

        test_image_urls = [
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/car.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/duck.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/fieldhockey.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/flower.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/runners.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/shark.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/soccer2.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/tree.jpg',
        ]

        for img in download_test_images(test_image_urls, tmp_path):
            img_data = normalize_image(img)
            mx_class_ids, mx_scores, mx_boxes = M.predict(img_data)
            onnx_scores, onnx_class_ids, onnx_boxes = session.run([], {input_name: img_data.asnumpy()})
            assert_almost_equal(mx_class_ids, onnx_class_ids)
            assert_almost_equal(mx_scores, onnx_scores)
            assert_almost_equal(mx_boxes, onnx_boxes)

    finally:
        shutil.rmtree(tmp_path)


@pytest.mark.parametrize('model', [
    'fcn_resnet50_ade',
    'fcn_resnet101_ade',
    'deeplab_resnet50_ade',
    'deeplab_resnet101_ade',
    'deeplab_resnest50_ade',
    'deeplab_resnest101_ade',
    'deeplab_resnest200_ade',
    'deeplab_resnest269_ade',
    'fcn_resnet101_coco',
    'deeplab_resnet101_coco',
    'fcn_resnet101_voc',
    'deeplab_resnet101_voc',
    'deeplab_resnet152_voc',
    'deeplab_resnet50_citys',
    'deeplab_resnet101_citys',
    'deeplab_v3b_plus_wideresnet_citys'
])
def test_img_segmentation_model_inference_onnxruntime(tmp_path, model):
    def normalize_image(imgfile):
        img = mx.image.imread(imgfile).astype('float32')
        img, _ = mx.image.center_crop(img, size=(480, 480))
        img = gluoncv.data.transforms.presets.segmentation.test_transform(img, mx.cpu(0))
        return img


    try:
        tmp_path = str(tmp_path)
        M = GluonModel(model, (1,3,480,480), 'float32', tmp_path)
        onnx_file = M.export_onnx()
        # create onnxruntime session using the generated onnx file
        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)
        input_name = session.get_inputs()[0].name

        test_image_urls = [
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/bikers.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/car.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/dancer.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/duck.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/fieldhockey.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/flower.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/runners.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/shark.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/soccer2.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/tree.jpg',
        ]

        for img in download_test_images(test_image_urls, tmp_path):
            img_data = normalize_image(img)
            mx_result = M.predict(img_data)
            onnx_result = session.run([], {input_name: img_data.asnumpy()})
            assert(len(mx_result) == len(onnx_result))
            for i in range(len(mx_result)):
                assert_almost_equal(mx_result[i], onnx_result[i])

    finally:
        shutil.rmtree(tmp_path)

@pytest.mark.parametrize('model', [
    'simple_pose_resnet18_v1b',
    'simple_pose_resnet50_v1b',
    'simple_pose_resnet50_v1d',
    'simple_pose_resnet101_v1b',
    'simple_pose_resnet101_v1d',
    'simple_pose_resnet152_v1b',
    'simple_pose_resnet152_v1d',
    'alpha_pose_resnet101_v1b_coco',
    #'mobile_pose_resnet18_v1b',
    #'mobile_pose_resnet50_v1b',
    #'mobile_pose_mobilenet1.0',
    #'mobile_pose_mobilenetv2_1.0',
    #'mobile_pose_mobilenetv3_large',
    #'mobile_pose_mobilenetv3_small',
])
def test_pose_estimation_model_inference_onnxruntime(tmp_path, model):
    def normalize_image(imgfile):
        img = mx.image.imread(imgfile).astype('float32')
        img, _ = mx.image.center_crop(img, size=(512, 512))
        img = gluoncv.data.transforms.presets.segmentation.test_transform(img, mx.cpu(0))
        return img

    try:
        tmp_path = str(tmp_path)
        M = GluonModel(model, (1,3,512,512), 'float32', tmp_path)
        onnx_file = M.export_onnx()
        # create onnxruntime session using the generated onnx file
        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)
        input_name = session.get_inputs()[0].name

        test_image_urls = [
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/bikers.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/dancer.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/fieldhockey.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/runners.jpg',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/soccer2.jpg',
        ]

        for img in download_test_images(test_image_urls, tmp_path):
            img_data = normalize_image(img)
            mx_result = M.predict(img_data)
            onnx_result = session.run([], {input_name: img_data.asnumpy()})
            assert(len(mx_result) == len(onnx_result))
            for i in range(len(mx_result)):
                assert_almost_equal(mx_result[i], onnx_result[i])

    finally:
        shutil.rmtree(tmp_path)


@pytest.mark.parametrize('model', [
    'inceptionv1_kinetics400',
    'resnet18_v1b_kinetics400',
    'resnet34_v1b_kinetics400',
    'resnet50_v1b_kinetics400',
    'resnet101_v1b_kinetics400',
    'resnet152_v1b_kinetics400',
    'resnet50_v1b_hmdb51',
    'resnet50_v1b_sthsthv2',
    'vgg16_ucf101',
    'inceptionv3_kinetics400',
    'inceptionv3_ucf101',
])
def test_action_recognition_model_inference_onnxruntime(tmp_path, model):
    batch_size = 64
    input_len = 224
    if 'inceptionv3' in model:
        input_len = 340

    def load_video(filepath):
        iterator = mx.image.ImageIter(batch_size=batch_size, data_shape=(3,input_len,input_len), path_imgrec=filepath)
        for batch in iterator:
            return batch.data[0]

    try:
        tmp_path = str(tmp_path)
        M = GluonModel(model, (batch_size,3,input_len,input_len), 'float32', tmp_path)
        onnx_file = M.export_onnx()
        # create onnxruntime session using the generated onnx file
        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)
        input_name = session.get_inputs()[0].name

        test_video_urls = [
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/actions/biking.rec',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/actions/diving.rec',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/actions/golfing.rec',
            'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/actions/sledding.rec',
        ]

        for video in download_test_images(test_video_urls, tmp_path):
            data = load_video(video)
            mx_result = M.predict(data)
            onnx_result = session.run([], {input_name: data.asnumpy()})[0]
            assert_almost_equal(mx_result, onnx_result, rtol=0.001, atol=0.01)

    finally:
        shutil.rmtree(tmp_path)

