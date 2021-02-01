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
        self.model = mx.gluon.model_zoo.vision.get_model(self.model_name, pretrained=True, ctx=self.ctx, root=self.modelpath)
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

@pytest.mark.parametrize('model', [
    'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'mobilenet1.0', 'mobilenet0.75', 'mobilenet0.5', 'mobilenet0.25',
    'mobilenetv2_1.0', 'mobilenetv2_0.75', 'mobilenetv2_0.5', 'mobilenetv2_0.25',
    'resnet18_v1', 'resnet18_v2', 'resnet34_v1', 'resnet34_v2', 'resnet50_v1', 'resnet50_v2',
    'resnet101_v1', 'resnet101_v2', 'resnet152_v1', 'resnet152_v2',
    'squeezenet1.0', 'squeezenet1.1',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
])
def test_obj_class_model_inference_onnxruntime(tmp_path, model):
    def normalize_image(imgfile):
        img_data = mx.image.imread(imgfile).transpose([2, 0, 1]).astype('float32')
        mean_vec = mx.nd.array([0.485, 0.456, 0.406])
        stddev_vec = mx.nd.array([0.229, 0.224, 0.225])
        norm_img_data = mx.nd.zeros(img_data.shape).astype('float32')
        for i in range(img_data.shape[0]):
            norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        return norm_img_data.reshape(1, 3, 224, 224).astype('float32')

    try:
        tmp_path = str(tmp_path)
        M = GluonModel(model, (1,3,224,224), 'float32', tmp_path)
        onnx_file = M.export_onnx()

        # create onnxruntime session using the generated onnx file
        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)
        input_name = session.get_inputs()[0].name

        test_image_urls = [
            'https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/onnx/images/dog.jpg',
            'https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/onnx/images/apron.jpg',
            'https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/onnx/images/dolphin.jpg',
            'https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/onnx/images/hammerheadshark.jpg',
            'https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/onnx/images/lotus.jpg'
        ]

        for img in download_test_images(test_image_urls, tmp_path):
            img_data = normalize_image(img)
            mx_result = M.predict(img_data)
            onnx_result = session.run([], {input_name: img_data.asnumpy()})[0]
            assert_almost_equal(mx_result, onnx_result)

    finally:
        shutil.rmtree(tmp_path)



class GluonCVModel(GluonModel):
    def __init__(self, *args, **kwargs):
        super(GluonCVModel, self).__init__(*args, **kwargs)
    def get_model(self):
        self.model = gluoncv.model_zoo.get_model(self.model_name, pretrained=True, ctx=self.ctx)
        self.model.hybridize()

@pytest.mark.parametrize('model', [
    'center_net_resnet18_v1b_voc',
    'center_net_resnet50_v1b_voc',
    'center_net_resnet101_v1b_voc',
    'center_net_resnet18_v1b_coco',
    'center_net_resnet50_v1b_coco',
    'center_net_resnet101_v1b_coco'
])
def test_obj_detection_model_inference_onnxruntime(tmp_path, model):
    def normalize_image(imgfile):
        x, _ = gluoncv.data.transforms.presets.center_net.load_test(imgfile, short=512)
        return x

    try:
        tmp_path = str(tmp_path)
        M = GluonCVModel(model, (1,3,512,683), 'float32', tmp_path)
        onnx_file = M.export_onnx()
        # create onnxruntime session using the generated onnx file
        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)
        input_name = session.get_inputs()[0].name

        test_image_urls = ['https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg']

        for img in download_test_images(test_image_urls, tmp_path):
            img_data = normalize_image(os.path.join(tmp_path, img))
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
    # the 4 models below are failing due to an accuracy issue
    #'deeplab_resnest50_ade',
    #'deeplab_resnest101_ade',
    #'deeplab_resnest200_ade',
    #'deeplab_resnest269_ade',
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
        img = mx.image.imresize(img, 480, 480)
        x = gluoncv.data.transforms.presets.segmentation.test_transform(img, mx.cpu(0))
        return x


    try:
        tmp_path = str(tmp_path)
        M = GluonCVModel(model, (1,3,480,480), 'float32', tmp_path)
        onnx_file = M.export_onnx()
        # create onnxruntime session using the generated onnx file
        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)
        input_name = session.get_inputs()[0].name

        test_image_urls = [
            'https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg',
            'https://cdn.cnn.com/cnnnext/dam/assets/201030094143-stock-rhodesian-ridgeback-super-tease.jpg'
        ]

        for img in download_test_images(test_image_urls, tmp_path):
            img_data = normalize_image(os.path.join(tmp_path, img))
            mx_result = M.predict(img_data)
            onnx_result = session.run([], {input_name: img_data.asnumpy()})
            assert(len(mx_result) == len(onnx_result))
            for i in range(len(mx_result)):
                assert_almost_equal(mx_result[i], onnx_result[i])

    finally:
        shutil.rmtree(tmp_path)



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

        assert_almost_equal(seq_encoding, pred_onx)
        assert_almost_equal(cls_encoding, cls_onx)

    finally:
        shutil.rmtree(tmp_path)


