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

    def export_onnx_dynamic(self, dynamic_input_shapes):
        onnx_file = self.modelpath + ".onnx"
        mx.contrib.onnx.export_model(self.modelpath + "-symbol.json", self.modelpath + "-0000.params",
                                     [self.input_shape], self.input_dtype, onnx_file, dynamic=True,
                                     dynamic_input_shapes=dynamic_input_shapes)
        return onnx_file

    def predict(self, data):
        return self.model(data)



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



@pytest.fixture(scope="session")
def obj_class_test_images(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("obj_class_data")
    from urllib.parse import urlparse
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
    paths = []
    for url in test_image_urls:
        fn = os.path.join(tmpdir, os.path.basename(urlparse(url).path))
        mx.test_utils.download(url, fname=fn)
        paths.append(fn)
    return paths

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
def test_obj_class_model_inference_onnxruntime(tmp_path, model, obj_class_test_images):
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

        for img in obj_class_test_images:
            img_data = normalize_image(img)
            mx_result = M.predict(img_data)
            onnx_result = session.run([], {input_name: img_data.asnumpy()})[0]
            assert_almost_equal(mx_result, onnx_result)

    finally:
        shutil.rmtree(tmp_path)


@pytest.fixture(scope="session")
def obj_detection_test_images(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("obj_det_data")
    from urllib.parse import urlparse
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
    paths = []
    for url in test_image_urls:
        fn = os.path.join(tmpdir, os.path.basename(urlparse(url).path))
        mx.test_utils.download(url, fname=fn)
        paths.append(fn)
    return paths


@pytest.mark.parametrize('model', [
    'center_net_resnet18_v1b_voc',
    'center_net_resnet50_v1b_voc',
    'center_net_resnet101_v1b_voc',
    'center_net_resnet18_v1b_coco',
    'center_net_resnet50_v1b_coco',
    'center_net_resnet101_v1b_coco',
    'ssd_300_vgg16_atrous_voc',
    'ssd_512_vgg16_atrous_voc',
    'ssd_512_resnet50_v1_voc',
    'ssd_512_mobilenet1.0_voc',
    'faster_rcnn_resnet50_v1b_voc',
    'yolo3_darknet53_voc',
    'yolo3_mobilenet1.0_voc',
    'ssd_300_vgg16_atrous_coco',
    'ssd_512_vgg16_atrous_coco',
    # 'ssd_300_resnet34_v1b_coco', #cannot import
    'ssd_512_resnet50_v1_coco',
    'ssd_512_mobilenet1.0_coco',
    'faster_rcnn_resnet50_v1b_coco',
    'faster_rcnn_resnet101_v1d_coco',
    'yolo3_darknet53_coco',
    'yolo3_mobilenet1.0_coco',
])
def test_obj_detection_model_inference_onnxruntime(tmp_path, model, obj_detection_test_images):
    def assert_obj_detetion_result(mx_ids, mx_scores, mx_boxes,
                                   onnx_ids, onnx_scores, onnx_boxes,
                                   score_thresh=0.6, score_tol=1e-4):
        def assert_bbox(mx_boxe, onnx_boxe, box_tol=1e-2):
            def assert_scalar(a, b, tol=box_tol):
                return np.abs(a-b) <= tol
            return assert_scalar(mx_boxe[0], onnx_boxe[0]) and assert_scalar(mx_boxe[1], onnx_boxe[1]) \
                      and assert_scalar(mx_boxe[2], onnx_boxe[2]) and assert_scalar(mx_boxe[3], onnx_boxe[3])

        found_match = False
        for i in range(len(onnx_ids)):
            onnx_id = onnx_ids[i][0]
            onnx_score = onnx_scores[i][0]
            onnx_boxe = onnx_boxes[i]

            if onnx_score < score_thresh:
                break
            for j in range(len(mx_ids)):
                mx_id = mx_ids[j].asnumpy()[0]
                mx_score = mx_scores[j].asnumpy()[0]
                mx_boxe = mx_boxes[j].asnumpy()
                # check socre 
                if onnx_score < mx_score - score_tol:
                    continue
                if onnx_score > mx_score + score_tol:
                    return False
                # check id
                if onnx_id != mx_id:
                    continue
                # check bounding box
                if assert_bbox(mx_boxe, onnx_boxe):
                    found_match = True
                    break
            if not found_match:
                return False
            found_match = False
        return True

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

        for img in obj_detection_test_images:
            img_data = normalize_image(img)
            mx_class_ids, mx_scores, mx_boxes = M.predict(img_data)
            # center_net_resnet models have different output format
            if 'center_net_resnet' in model:
                onnx_scores, onnx_class_ids, onnx_boxes = session.run([], {input_name: img_data.asnumpy()})
                assert_almost_equal(mx_class_ids, onnx_class_ids)
                assert_almost_equal(mx_scores, onnx_scores)
                assert_almost_equal(mx_boxes, onnx_boxes)
            else:
                onnx_class_ids, onnx_scores, onnx_boxes = session.run([], {input_name: img_data.asnumpy()})
                if not assert_obj_detetion_result(mx_class_ids[0], mx_scores[0], mx_boxes[0], \
                        onnx_class_ids[0], onnx_scores[0], onnx_boxes[0]):
                    raise AssertionError("Assertion error on model: " + model)

    finally:
        shutil.rmtree(tmp_path)

@pytest.fixture(scope="session")
def img_segmentation_test_images(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("img_seg_data")
    from urllib.parse import urlparse
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
    paths = []
    for url in test_image_urls:
        fn = os.path.join(tmpdir, os.path.basename(urlparse(url).path))
        mx.test_utils.download(url, fname=fn)
        paths.append(fn)
    return paths

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
def test_img_segmentation_model_inference_onnxruntime(tmp_path, model, img_segmentation_test_images):
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

        for img in img_segmentation_test_images:
            img_data = normalize_image(img)
            mx_result = M.predict(img_data)
            onnx_result = session.run([], {input_name: img_data.asnumpy()})
            assert(len(mx_result) == len(onnx_result))
            for i in range(len(mx_result)):
                assert_almost_equal(mx_result[i], onnx_result[i])

    finally:
        shutil.rmtree(tmp_path)


@pytest.fixture(scope="session")
def pose_estimation_test_images(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("pose_est_data")
    from urllib.parse import urlparse
    test_image_urls = [
        'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/bikers.jpg',
        'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/dancer.jpg',
        'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/fieldhockey.jpg',
        'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/runners.jpg',
        'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/images/soccer2.jpg',
    ]
    paths = []
    for url in test_image_urls:
        fn = os.path.join(tmpdir, os.path.basename(urlparse(url).path))
        mx.test_utils.download(url, fname=fn)
        paths.append(fn)
    return paths

@pytest.mark.parametrize('model', [
    'simple_pose_resnet18_v1b',
    'simple_pose_resnet50_v1b',
    'simple_pose_resnet50_v1d',
    'simple_pose_resnet101_v1b',
    'simple_pose_resnet101_v1d',
    'simple_pose_resnet152_v1b',
    'simple_pose_resnet152_v1d',
    'alpha_pose_resnet101_v1b_coco',
    'mobile_pose_resnet18_v1b',
    'mobile_pose_resnet50_v1b',
    'mobile_pose_mobilenet1.0',
    'mobile_pose_mobilenetv2_1.0',
    'mobile_pose_mobilenetv3_large',
    'mobile_pose_mobilenetv3_small',
])
def test_pose_estimation_model_inference_onnxruntime(tmp_path, model, pose_estimation_test_images):
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

        for img in pose_estimation_test_images:
            img_data = normalize_image(img)
            mx_result = M.predict(img_data)
            onnx_result = session.run([], {input_name: img_data.asnumpy()})
            assert(len(mx_result) == len(onnx_result))
            for i in range(len(mx_result)):
                assert_almost_equal(mx_result[i], onnx_result[i])

    finally:
        shutil.rmtree(tmp_path)

@pytest.fixture(scope="session")
def act_recognition_test_data(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("act_rec_data")
    from urllib.parse import urlparse
    test_image_urls = [
        'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/actions/biking.rec',
        'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/actions/diving.rec',
        'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/actions/golfing.rec',
        'https://github.com/apache/incubator-mxnet-ci/raw/master/test-data/actions/sledding.rec',
    ]
    paths = []
    for url in test_image_urls:
        fn = os.path.join(tmpdir, os.path.basename(urlparse(url).path))
        mx.test_utils.download(url, fname=fn)
        paths.append(fn)
    return paths

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
def test_action_recognition_model_inference_onnxruntime(tmp_path, model, act_recognition_test_data):
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

        for video in act_recognition_test_data:
            data = load_video(video)
            mx_result = M.predict(data)
            onnx_result = session.run([], {input_name: data.asnumpy()})[0]
            assert_almost_equal(mx_result, onnx_result, rtol=0.001, atol=0.01)

    finally:
        shutil.rmtree(tmp_path)


@with_seed()
@pytest.mark.parametrize('model_name', ['roberta_24_1024_16', 'roberta_12_768_12'])
def test_roberta_inference_onnxruntime(tmp_path, model_name):
    tmp_path = str(tmp_path)
    try:
        import gluonnlp as nlp
        ctx = mx.cpu(0)

        dataset= 'openwebtext_ccnews_stories_books_cased'#'book_corpus_wiki_en_uncased'
        model, _ = nlp.model.get_model(
        name=model_name,
        ctx=ctx,
        pretrained=True,
        use_decoder=True,
        dataset_name=dataset)
        
        model.hybridize(static_alloc=False)

        batch = 2
        seq_length = 32
        num_masked_positions = 1
        inputs = mx.nd.random.uniform(0, 30522, shape=(batch, seq_length), dtype='float32', ctx=ctx)
        valid_length = mx.nd.array([seq_length] * batch, dtype='float32', ctx=ctx)
        masked_positions = mx.nd.random.uniform(0, 32, shape=(batch, num_masked_positions),
            dtype='float32', ctx=ctx).astype('int32')

        sequence_outputs, attention_outputs= model(inputs, valid_length, masked_positions)    

        prefix = "%s/roberta" % tmp_path
        model.export(prefix)

        sym_file = "%s-symbol.json" % prefix
        params_file = "%s-0000.params" % prefix
        onnx_file = "%s.onnx" % prefix
        input_shapes = [(batch, seq_length), (batch,), (batch, num_masked_positions)]
        converted_model_path = mx.contrib.onnx.export_model(sym_file, params_file, input_shapes,
                                                            [np.float32, np.float32, np.int32],
                                                            onnx_file, verbose=True)

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = onnxruntime.InferenceSession(onnx_file, sess_options)

        in_tensors = [inputs, valid_length, masked_positions]
        input_dict = dict((sess.get_inputs()[i].name, in_tensors[i].asnumpy()) for i in range(len(in_tensors)))
        pred = sess.run(None, input_dict)

        assert_almost_equal(sequence_outputs, pred[0])
        assert_almost_equal(attention_outputs, pred[1])

    finally:
        shutil.rmtree(tmp_path)


@with_seed()
@pytest.mark.parametrize('model', ['bert_12_768_12', 'bert_24_1024_16'])
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
            pretrained=True,
            use_pooler=True,
            use_decoder=False,
            use_classifier=False)

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
        input_types = [np.float32, np.float32, np.float32]
        converted_model_path = mx.contrib.onnx.export_model(sym_file, params_file, input_shapes, input_types, onnx_file)


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


@with_seed()
@pytest.mark.parametrize('model_name', ['distilbert_6_768_12'])
def test_distilbert_inference_onnxruntime(tmp_path, model_name):
    tmp_path = str(tmp_path)
    try:
        import gluonnlp as nlp
        dataset = 'distilbert_book_corpus_wiki_en_uncased'
        ctx = mx.cpu(0)
        model, _ = nlp.model.get_model(
            name=model_name,
            ctx=ctx,
            pretrained=True,
            dataset_name=dataset)

        model.hybridize(static_alloc=True)

        batch = 2
        seq_length = 32
        num_masked_positions = 1
        inputs = mx.nd.random.uniform(0, 30522, shape=(batch, seq_length), dtype='float32', ctx=ctx)
        valid_length = mx.nd.array([seq_length] * batch, dtype='float32', ctx=ctx)

        sequence_outputs = model(inputs, valid_length)

        prefix = "%s/distilbert" % tmp_path
        model.export(prefix)
        sym_file = "%s-symbol.json" % prefix
        params_file = "%s-0000.params" % prefix
        onnx_file = "%s.onnx" % prefix

        input_shapes = [(batch, seq_length), (batch,)]
        converted_model_path = mx.contrib.onnx.export_model(sym_file, params_file, input_shapes,
                                                            [np.float32, np.float32],
                                                            onnx_file, verbose=True)
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = onnxruntime.InferenceSession(onnx_file, sess_options)

        in_tensors = [inputs, valid_length]
        input_dict = dict((sess.get_inputs()[i].name, in_tensors[i].asnumpy()) for i in range(len(in_tensors)))
        pred = sess.run(None, input_dict)

        assert_almost_equal(sequence_outputs, pred[0])

    finally:
        shutil.rmtree(tmp_path)


@with_seed()
@pytest.mark.parametrize('model_name', [('standard_lstm_lm_200', 200), ('standard_lstm_lm_650', 650),
                                        ('standard_lstm_lm_1500', 1500)])
@pytest.mark.parametrize('seq_length', [16, 32])
def test_standard_rnn_lstm_pretrained_inference_onnxruntime(tmp_path, model_name, seq_length):
    try:
        import gluonnlp as nlp
        ctx = mx.cpu()
        dataset= 'wikitext-2'
        model, _ = nlp.model.get_model(
            name=model_name[0],
            ctx=ctx,
            pretrained=True,
            dataset_name=dataset,
            dropout=0)
        model.hybridize()

        batch = 2
        num_hidden = model_name[1]
        num_layers = 2
        inputs = mx.nd.random.randint(0, 33278, shape=(seq_length, batch),
                                      ctx=ctx).astype('float32')
        begin_state = model.begin_state(func=mx.nd.random.uniform, low=0, high=1,
                                        batch_size=batch, dtype='float32', ctx=ctx)
        out, out_state= model(inputs, begin_state)

        prefix = "%s/standard_rnn_lstm" % tmp_path
        model.export(prefix)
        sym_file = "%s-symbol.json" % prefix
        params_file = "%s-0000.params" % prefix
        onnx_file = "%s.onnx" % prefix

        input_shapes = [(seq_length, batch), np.shape(begin_state[0]), np.shape(begin_state[1])]
        converted_model_path = mx.contrib.onnx.export_model(sym_file, params_file, input_shapes,
                                                            [np.float32, np.float32, np.float32],
                                                            onnx_file, verbose=True)
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = onnxruntime.InferenceSession(onnx_file, sess_options)

        in_tensors = [inputs, begin_state[0], begin_state[1]]
        input_dict = dict((sess.get_inputs()[i].name, in_tensors[i].asnumpy()) for i in range(len(in_tensors)))
        pred = sess.run(None, input_dict)

        assert_almost_equal(out, pred[2])
        assert_almost_equal(out_state[0], pred[0])
        assert_almost_equal(out_state[1], pred[1])

    finally:
        shutil.rmtree(tmp_path)


@with_seed()
@pytest.mark.parametrize('model_name', ['mobilenet1.0', 'inceptionv3', 'darknet53', 'resnest14'])
def test_dynamic_shape_cv_inference_onnxruntime(tmp_path, model_name):
    tmp_path = str(tmp_path)
    try:
        M = GluonModel(model_name, (1, 3, 512, 512), 'float32', tmp_path)
        dynamic_input_shapes = [(None, 3, 512, 512)]
        onnx_file = M.export_onnx_dynamic(dynamic_input_shapes)

        # create onnxruntime session using the generated onnx file
        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        sess = onnxruntime.InferenceSession(onnx_file, ses_opt)

        # test on a different batch size
        x = mx.random.uniform(0, 10, (5, 3, 512, 512))
        in_tensors = [x]
        input_dict = dict((sess.get_inputs()[i].name, in_tensors[i].asnumpy()) for i in range(len(in_tensors)))
        pred_on = sess.run(None, input_dict)

        pred_mx = M.predict(x)

        assert_almost_equal(pred_mx, pred_on[0])

    finally:
        shutil.rmtree(tmp_path)


@with_seed()
@pytest.mark.parametrize('model', ['bert_12_768_12'])
def test_dynamic_shape_bert_inference_onnxruntime(tmp_path, model):
    tmp_path = str(tmp_path)
    try:
        import gluonnlp as nlp
        dataset = 'book_corpus_wiki_en_uncased'
        ctx = mx.cpu(0)
        model, vocab = nlp.model.get_model(
            name=model,
            ctx=ctx,
            dataset_name=dataset,
            pretrained=True,
            use_pooler=True,
            use_decoder=False,
            num_layers = 3,
            hparam_allow_override = True,
            use_classifier=False)

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

        dynamic_input_shapes = [(None, seq_length), (None, seq_length), (None,)]
        input_shapes = [(batch, seq_length), (batch, seq_length), (batch,)]
        input_types = [np.float32, np.float32, np.float32]
        converted_model_path = mx.contrib.onnx.export_model(sym_file, params_file, input_shapes,
                                                            input_types, onnx_file,
                                                            dynamic=True,
                                                            dynamic_input_shapes=dynamic_input_shapes)

        # create onnxruntime session using the generated onnx file
        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)

        # test on a different batch size
        batch = 7
        seq_length = 16
        inputs = mx.nd.random.uniform(0, 30522, shape=(batch, seq_length), dtype='float32')
        token_types = mx.nd.random.uniform(0, 2, shape=(batch, seq_length), dtype='float32')
        valid_length = mx.nd.array([seq_length] * batch, dtype='float32')

        seq_encoding, cls_encoding = model(inputs, token_types, valid_length)

        onnx_inputs = [inputs, token_types, valid_length]
        input_dict = dict((session.get_inputs()[i].name, onnx_inputs[i].asnumpy()) for i in range(len(onnx_inputs)))
        pred_onx, cls_onx = session.run(None, input_dict)

        assert_almost_equal(seq_encoding, pred_onx)
        assert_almost_equal(cls_encoding, cls_onx)

    finally:
        shutil.rmtree(tmp_path)


@with_seed()
@pytest.mark.parametrize('model_name', [('awd_lstm_lm_600', 600), ('awd_lstm_lm_1150', 1150)])
@pytest.mark.parametrize('seq_length', [16, 128, 256])
def test_awd_rnn_lstm_pretrained_inference_onnxruntime(tmp_path, model_name, seq_length):
    try:
        import gluonnlp as nlp
        ctx = mx.cpu()
        dataset= 'wikitext-2'
        model, _ = nlp.model.get_model(
            name=model_name[0],
            ctx=ctx,
            pretrained=True,
            dataset_name=dataset,
            dropout=0)
        model.hybridize()

        batch = 2
        num_hidden = model_name[1]
        num_layers = 2
        inputs = mx.nd.random.randint(0, 33278, shape=(seq_length, batch),
                                      ctx=ctx).astype('float32')
        begin_state = model.begin_state(func=mx.nd.random.uniform, low=0, high=1,
                                        batch_size=batch, dtype='float32', ctx=ctx)
        out, out_state= model(inputs, begin_state)

        prefix = "%s/awd_lstm" % tmp_path
        model.export(prefix)
        sym_file = "%s-symbol.json" % prefix
        params_file = "%s-0000.params" % prefix
        onnx_file = "%s.onnx" % prefix

        input_shapes = [(seq_length, batch), 
                        np.shape(begin_state[0][0]), np.shape(begin_state[0][1]),
                        np.shape(begin_state[1][0]), np.shape(begin_state[1][1]),
                        np.shape(begin_state[2][0]), np.shape(begin_state[2][1])]
        input_types = [np.float32, np.float32, np.float32, np.float32, np.float32, np.float32,
                       np.float32]
        converted_model_path = mx.contrib.onnx.export_model(sym_file, params_file, input_shapes,
                                                            input_types, onnx_file, verbose=True)

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = onnxruntime.InferenceSession(onnx_file, sess_options)

        in_tensors = [inputs, begin_state[0][0], begin_state[0][1],
                      begin_state[1][0], begin_state[1][1],
                      begin_state[2][0], begin_state[2][1]]
        input_dict = dict((sess.get_inputs()[i].name, in_tensors[i].asnumpy()) for i in range(len(in_tensors)))
        pred = sess.run(None, input_dict)

        assert_almost_equal(out, pred[6])
        assert_almost_equal(out_state[0][0], pred[0])
        assert_almost_equal(out_state[0][1], pred[1])
        assert_almost_equal(out_state[1][0], pred[2])
        assert_almost_equal(out_state[1][1], pred[3])
        assert_almost_equal(out_state[2][0], pred[4])
        assert_almost_equal(out_state[2][1], pred[5])

    finally:
        shutil.rmtree(tmp_path)


@with_seed()
@pytest.mark.parametrize('model_name', ['ernie_12_768_12'])
def test_ernie_inference_onnxruntime(tmp_path, model_name):
    tmp_path = str(tmp_path)
    try:
        import gluonnlp as nlp
        dataset = 'baidu_ernie_uncased'
        ctx = mx.cpu(0)
        model, vocab = nlp.model.get_model(
            name=model_name,
            ctx=ctx,
            dataset_name=dataset,
            pretrained=True,
            use_pooler=True,
            use_decoder=False,
            num_layers = 3,
            hparam_allow_override = True,
            use_classifier=False)

        model.hybridize(static_alloc=True)

        batch = 5
        seq_length = 16
        # create synthetic test data
        inputs = mx.nd.random.uniform(0, 17964, shape=(batch, seq_length), dtype='float32')
        token_types = mx.nd.random.uniform(0, 2, shape=(batch, seq_length), dtype='float32')
        valid_length = mx.nd.array([seq_length] * batch, dtype='float32')

        seq_encoding, cls_encoding = model(inputs, token_types, valid_length)

        prefix = "%s/ernie" % tmp_path
        model.export(prefix)
        sym_file = "%s-symbol.json" % prefix
        params_file = "%s-0000.params" % prefix
        onnx_file = "%s.onnx" % prefix

        input_shapes = [(batch, seq_length), (batch, seq_length), (batch,)]
        input_types = [np.float32, np.float32, np.float32]
        converted_model_path = mx.contrib.onnx.export_model(sym_file, params_file, input_shapes,
                                                            input_types, onnx_file)

        # create onnxruntime session using the generated onnx file
        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)

        seq_encoding, cls_encoding = model(inputs, token_types, valid_length)

        onnx_inputs = [inputs, token_types, valid_length]
        input_dict = dict((session.get_inputs()[i].name, onnx_inputs[i].asnumpy()) for i in range(len(onnx_inputs)))
        pred_onx, cls_onx = session.run(None, input_dict)

        assert_almost_equal(seq_encoding, pred_onx)
        assert_almost_equal(cls_encoding, cls_onx)

    finally:
        shutil.rmtree(tmp_path)


@with_seed()
@pytest.mark.parametrize('model_name', ['transformer_en_de_512'])
def test_transformer_pretrained_inference_onnxruntime(tmp_path, model_name):
    tmp_path = str(tmp_path)
    try:
        import gluonnlp as nlp
        dataset = 'WMT2014'
        ctx = mx.cpu(0)
        model, _, _ = nlp.model.get_model(
            name=model_name,
            ctx=ctx,
            pretrained=True,
            dataset_name=dataset)

        model.hybridize(static_alloc=False)

        batch = 7
        seq_length = 16
        C_in = 512
        C_out = 512
        src = mx.nd.random.uniform(0, 36794, shape=(batch, seq_length), dtype='float32')
        step_input = mx.nd.random.uniform(0, 36794, shape=(batch,), dtype='float32')
        src_valid_length = mx.nd.array([seq_length] * batch, dtype='float32')

        encoder_outputs, encoder_additional_outputs = model.encode(src,
                                                                   valid_length=src_valid_length)

        decoder_states = model.decoder.init_state_from_encoder(encoder_outputs, src_valid_length)

        step_output, states, additional_outputs = model.decode_step(step_input, decoder_states)

        # skip export of 'decoder' as it's used for training only
        for component in ['encoder', 'one_step_ahead_decoder', 'src_embed', 'tgt_embed',
                         'tgt_proj']:

            prefix = "%s/%s" %(tmp_path, component)
            component = getattr(model, component)
            component.export(prefix)
            sym_file = "%s-symbol.json" % prefix
            params_file = "%s-0000.params" % prefix
            onnx_file = "%s.onnx" % prefix

        def export_to_onnx(prefix, input_shapes, input_types, **kwargs):
            sym_file = "%s-symbol.json" % prefix
            params_file = "%s-0000.params" % prefix
            onnx_file = "%s.onnx" % prefix
            return mx.contrib.onnx.export_model(sym_file, params_file, input_shapes, input_types,
                                                onnx_file, **kwargs)

        def onnx_runtime_predict(onnx_file, onnx_inputs):
            ses_opt = onnxruntime.SessionOptions()
            ses_opt.log_severity_level = 3
            session = onnxruntime.InferenceSession(onnx_file, ses_opt)
            input_dict = dict((session.get_inputs()[i].name, onnx_inputs[i].asnumpy())
                            for i in range(len(onnx_inputs)))
            return session.run(None, input_dict)

        def verify_encoder():
            inputs = mx.nd.random.uniform(-1, 1, shape=(batch, seq_length, C_in), dtype='float32')
            valid_length = mx.nd.array([seq_length] * batch, dtype='float32')
            pred = model.encoder(inputs, valid_length=valid_length)

            prefix = "%s/encoder" %tmp_path
            input_shapes = [(batch, seq_length, C_in), (batch,)]
            input_types = [np.float32, np.float32]
            onnx_file = export_to_onnx(prefix, input_shapes, input_types)
            onnx_inputs = [inputs, valid_length]
            pred_onx = onnx_runtime_predict(onnx_file, onnx_inputs)

            assert_almost_equal(pred[0], pred_onx[0])

        def verify_src_embed():
            src = mx.nd.random.uniform(0, 36794, shape=(batch, seq_length), dtype='float32')
            pred = model.src_embed(src)

            prefix = "%s/src_embed" %tmp_path
            input_shapes = [(batch, seq_length)]
            input_types = [np.float32]
            onnx_file = export_to_onnx(prefix, input_shapes, input_types)
            onnx_inputs = [src]
            pred_onx = onnx_runtime_predict(onnx_file, onnx_inputs)

            assert_almost_equal(pred, pred_onx[0])

        def verify_tgt_embed():
            tgt = mx.nd.random.uniform(0, 36794, shape=(batch, seq_length), dtype='float32')
            pred = model.tgt_embed(tgt)

            prefix = "%s/tgt_embed" %tmp_path
            input_shapes = [(batch, seq_length)]
            input_types = [np.float32]
            onnx_file = export_to_onnx(prefix, input_shapes, input_types)
            onnx_inputs = [tgt]
            pred_onx = onnx_runtime_predict(onnx_file, onnx_inputs)

            assert_almost_equal(pred, pred_onx[0])

        def verify_tgt_proj():
            decoder_out = mx.nd.random.uniform(0, 512, shape=(batch, seq_length, C_out),
                                               dtype='float32')
            pred = model.tgt_proj(decoder_out)

            prefix = "%s/tgt_proj" %tmp_path
            input_shapes = [(batch, seq_length, C_out)]
            input_types = [np.float32]
            onnx_file = export_to_onnx(prefix, input_shapes, input_types)
            onnx_inputs = [decoder_out]
            pred_onx = onnx_runtime_predict(onnx_file, onnx_inputs)

            assert_almost_equal(pred, pred_onx[0], rtol=1.e-04, atol=1.5e-03)

        def verify_one_step_ahead_decoder():
            prefix = "%s/one_step_ahead_decoder" %tmp_path

            # the input data order
            perm = [2, 0, 1]
            input_shapes = [(batch, seq_length, C_in), (batch, seq_length, C_out),
                            (batch, seq_length)]
            input_shapes = [input_shapes[i] for i in perm]
            dynamic_input_shapes = [(batch, 'seq_length', C_in), (batch, 'seq_length', C_out),
                                    (batch, 'seq_length')]
            dynamic_input_shapes = [dynamic_input_shapes[i] for i in perm]
            input_types = [np.float32, np.float32, np.float32]
            # do a dynamic export
            onnx_file = export_to_onnx(prefix, input_shapes, input_types, dynamic=True,
                                       dynamic_input_shapes=dynamic_input_shapes)

            # step 0
            step_input = mx.nd.random.uniform(-1, 1, shape=(batch, C_in), dtype='float32')
            # mxnet
            pred, step_states, _ = model.one_step_ahead_decoder(step_input, decoder_states)
            # onnx
            # note that we need to expand the sequence axis just like in here:
            # https://github.com/dmlc/gluon-nlp/blob/v0.10.x/src/gluonnlp/model/transformer.py#L831
            input_onx = mx.nd.expand_dims(step_input, axis=1)
            onnx_inputs = [input_onx, decoder_states[0], decoder_states[1]]
            onnx_inputs = [onnx_inputs[i] for i in perm]
            pred_onx = onnx_runtime_predict(onnx_file, onnx_inputs)

            assert_almost_equal(pred, pred_onx[0])

            # step >= 1
            for i in range(20):
                step_input = mx.nd.random.uniform(-10*i, 10*i, shape=(batch, C_in), dtype='float32')
                # mxnet
                pred, step_states, _ = model.one_step_ahead_decoder(step_input, step_states)
                # onnx
                # note that we need to concat the step_input with the previous inpus
                # just like in here:
                # https://github.com/dmlc/gluon-nlp/blob/v0.10.x/src/gluonnlp/model/transformer.py#L828
                input_onx = mx.nd.concat(input_onx, mx.nd.expand_dims(step_input, axis=1), dim=1)
                onnx_inputs = [input_onx, decoder_states[0], decoder_states[1]]
                onnx_inputs = [onnx_inputs[i] for i in perm]
                pred_onx = onnx_runtime_predict(onnx_file, onnx_inputs)

                assert_almost_equal(pred, pred_onx[0])

        verify_encoder()
        verify_src_embed()
        verify_tgt_embed()
        verify_tgt_proj()
        verify_one_step_ahead_decoder()

    finally:
        shutil.rmtree(tmp_path)
