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
import onnxruntime

import json
import os
import shutil
import tempfile


def test_cv_model_inference_onnxruntime():
    def get_gluon_cv_model(model_name, tmp):
        tmpfile = os.path.join(tmp, model_name)
        ctx = mx.cpu(0)
        net_fp32 = mx.gluon.model_zoo.vision.get_model(model_name, pretrained=True, ctx=ctx, root=tmp)
        net_fp32.hybridize()
        data = mx.nd.zeros((1,3,224,224), dtype='float32', ctx=ctx)
        net_fp32.forward(data)
        net_fp32.export(tmpfile, 0)
        sym_file = tmpfile + '-symbol.json'
        params_file = tmpfile + '-0000.params'
        return sym_file, params_file

    def export_model_to_onnx(sym_file, params_file):
        input_shape = (1,3,224,224)
        onnx_file = os.path.join(os.path.dirname(sym_file), "model.onnx")
        converted_model_path = mx.contrib.onnx.export_model(sym_file, params_file, [input_shape],
                                                            np.float32, onnx_file)
        return onnx_file

    def normalize_image(imgfile):
        image = mx.image.imread(imgfile).asnumpy()
        image_data = np.array(image).transpose(2, 0, 1)
        img_data = image_data.astype('float32')
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(img_data.shape).astype('float32')
        for i in range(img_data.shape[0]):
            norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        return norm_img_data.reshape(1, 3, 224, 224).astype('float32')

    def get_prediction(model, image):
        pass

    def softmax(x):
        x = x.reshape(-1)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def load_imgnet_labels():
        mx.test_utils.download('https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/onnx/image_net_labels.json')
        return np.array(json.load(open('image_net_labels.json', 'r')))

    def download_test_images():
        test_images = [
            ['dog.jpg',['boxer']],
            ['apron.jpg', ['apron', 'maillot']],
            ['dolphin.jpg', ['great white shark','grey whale']],
            ['hammerheadshark.jpg', ['tiger shark']],
            ['lotus.jpg', ['pinwheel','pot']]
        ]
        for f,_ in test_images:
            mx.test_utils.download('https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/onnx/images/'+f+'?raw=true',
                                   fname=f)
        return test_images


    test_models = [
        'mobilenet1.0', 'mobilenet0.75', 'mobilenet0.5', 'mobilenet0.25',
        'mobilenetv2_1.0', 'mobilenetv2_0.75', 'mobilenetv2_0.5', 'mobilenetv2_0.25',
        'resnet18_v1', 'resnet18_v2', 'resnet34_v1', 'resnet34_v2', 'resnet50_v1', 'resnet50_v2',
        'resnet101_v1', 'resnet101_v2', 'resnet152_v1', 'resnet152_v2',
        'squeezenet1.0', 'squeezenet1.1', 
        'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
    ]
    labels = load_imgnet_labels()
    test_images = download_test_images()

    for model in test_models:
        tmpdir = tempfile.mkdtemp()
        sym_file, params_file = get_gluon_cv_model(model, tmpdir)
        onnx_file = export_model_to_onnx(sym_file, params_file)
        #print("exported onnx file: ",onnx_file)

        # create onnxruntime session using the generated onnx file
        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)
        input_name = session.get_inputs()[0].name

        for img,classes in test_images:
            img_data = normalize_image(img)
            raw_result = session.run([], {input_name: img_data})
            res = softmax(np.array(raw_result)).tolist()
            class_idx = np.argmax(res)
            #print("Image top classification:",labels[class_idx])
            sort_idx = np.flip(np.squeeze(np.argsort(res)))
            #print("\tTop labels: " + ",".join(labels[sort_idx[:5]]))
            correct_classification = False
            for label in labels[sort_idx[:5]]:
                for c in classes:
                    if c in label:
                        correct_classification = True
            assert correct_classification == True

        # cleanup
        shutil.rmtree(tmpdir)




if __name__ == "__main__":
    test_cv_model_inference_onnxruntime()

