"""
Description : Training module for LipNet
"""
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

import sys
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon.data.vision import transforms
from tqdm import tqdm, trange
from data_loader import LipsDataset
from models.network import LipNet
from BeamSearch import ctcBeamSearch
from utils.common import char_conv, int2char
# set gpu count
def setting_ctx(use_gpu):
    """
    Description : set gpu module
    """
    if use_gpu:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()
    return ctx


ALPHABET = ''
for i in range(27):
    ALPHABET += int2char(i)

def char_beam_search(out):
    """
    Description : apply beam search for prediction result
    """
    out_conv = list()
    for idx in range(out.shape[0]):
        probs = out[idx]
        prob = probs.softmax().asnumpy()
        line_string_proposals = ctcBeamSearch(prob, ALPHABET, None, k=4, beamWidth=25)
        out_conv.append(line_string_proposals[0])
    return out_conv
# pylint: disable=too-many-instance-attributes
class Train(object):
    """
    Description : Train class for training network
    """
    def __init__(self, config):
        ##setting hyper-parameters
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.image_path = config.image_path
        self.align_path = config.align_path
        self.dr_rate = config.dr_rate
        self.use_gpu = config.use_gpu
        self.ctx = setting_ctx(self.use_gpu)
        self.num_workers = config.num_workers
        self.build_model()
    def build_model(self):
        """
        Description : build network
        """
        #set network
        self.net = LipNet(self.dr_rate)
        self.net.initialize(ctx=self.ctx)
        #set optimizer
        self.loss_fn = gluon.loss.CTCLoss()
        self.trainer = gluon.Trainer(self.net.collect_params(), \
                                     optimizer='adam', \
                                     optimizer_params={'learning_rate':1e4,
                                                       'beta1':0.9,
                                                       'beta2':0.999})
    def save_model(self, epoch, iter_no, current_loss):
        """
        Description : save parameter of network weight
        """
        file_name = "checkpoint/best_model_epoches_"+str(epoch)+"iter_"+str(iter_no)+ \
        "loss_"+str(round(current_loss, 2))
        self.net.save_parameters(file_name)
    def train(self):
        """
        Description : training for LipNet
        """
        input_transform = transforms.Compose([transforms.ToTensor(),\
                                             transforms.Normalize((0.7136, 0.4906, 0.3283),
                                                                  (0.1138, 0.1078, 0.0917))])
        training_dataset = LipsDataset(self.image_path, self.align_path, transform=input_transform)
        train_dataloader = mx.gluon.data.DataLoader(training_dataset,
                                                    batch_size=self.batch_size, shuffle=True,
                                                    num_workers=self.num_workers)
        best_loss = sys.maxsize
        for epoch in trange(self.epochs):
            iter_no = 0
            for input_data, label in tqdm(train_dataloader):
                input_data = nd.transpose(input_data, (0, 2, 1, 3, 4))
                # pylint: disable=no-member
                input_data = input_data.copyto(self.ctx)
                label = label.copyto(self.ctx)
                with autograd.record():
                    with autograd.train_mode():
                        out = self.net(input_data)
                        loss_val = self.loss_fn(out, label)
                        loss_val.backward()
                self.trainer.step(input_data.shape[0])
                if iter_no % 20 == 0:
                    print("epoch:{e} iter:{i} loss:{l}".format(e=epoch,
                                                               i=iter_no,
                                                               l=loss_val.mean().asscalar()))
                    self.infer(input_data, label)
                    current_loss = loss_val.mean().asscalar()
                    if best_loss > current_loss:
                        self.save_model(epoch, iter_no, current_loss)
                        best_loss = current_loss
                iter_no += 1
    def infer(self, input_data, label):
        """
        Description : Print sentence for prediction result
        """
        pred = self.net(input_data)
        pred_convert = char_beam_search(pred)
        label_convert = char_conv(label.asnumpy())
        for target, pred in zip(label_convert, pred_convert):
            print("target:{t}  pred:{p}".format(t=target, p=pred))
