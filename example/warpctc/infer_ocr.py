# coding=utf-8
# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys

sys.path.insert(0, "../../python")
from __future__ import print_function
import numpy as np
import mxnet as mx

from lstm_model import LSTMInferenceModel

import cv2, random
from captcha.image import ImageCaptcha

BATCH_SIZE = 32
SEQ_LENGTH = 80


def ctc_label(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i + 1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret


def remove_blank(l):
    ret = []
    for i in range(len(l)):
        if l[i] == 0:
            break
        ret.append(l[i])
    return ret


def gen_rand():
    buf = ""
    max_len = random.randint(3,4)
    for i in range(max_len):
        buf += str(random.randint(0,9))
    return buf

if __name__ == '__main__':
    num_hidden = 100
    num_lstm_layer = 2

    num_epoch = 10
    learning_rate = 0.001
    momentum = 0.9
    num_label = 4

    n_channel = 1
    contexts = [mx.context.gpu(0)]
    _, arg_params, __ = mx.model.load_checkpoint('ocr', num_epoch)

    num = gen_rand()
    print('Generated number: ' + num)
    # change the fonts accordingly
    captcha = ImageCaptcha(fonts=['./data/OpenSans-Regular.ttf'])
    img = captcha.generate(num)
    img = np.fromstring(img.getvalue(), dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (80, 30))

    img = img.transpose(1, 0)

    img = img.reshape((1, 80 * 30))
    img = np.multiply(img, 1 / 255.0)

    data_shape = [('data', (1, n_channel * 80 * 30))]
    input_shapes = dict(data_shape)

    model = LSTMInferenceModel(num_lstm_layer,
                               SEQ_LENGTH,
                               num_hidden=num_hidden,
                               num_label=num_label,
                               arg_params=arg_params,
                               data_size = n_channel * 30 * 80,
                               ctx=contexts[0])

    prob = model.forward(mx.nd.array(img))

    p = []
    for k in range(SEQ_LENGTH):
        p.append(np.argmax(prob[k]))

    p = ctc_label(p)
    print('Predicted label: ' + str(p))

    pred = ''
    for c in p:
        pred += str((int(c) - 1))

    print('Predicted number: ' + pred)


