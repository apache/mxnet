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

"""
Description : Training module for LipNet
"""


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


def setting_ctx(num_gpus):
    """
    Description : set gpu module
    """
    if num_gpus > 0:
        ctx = [mx.gpu(i) for i in range(num_gpus)]
    else:
        ctx = [mx.cpu()]
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

# pylint: disable=too-many-instance-attributes, too-many-locals
class Train:
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
        self.num_gpus = config.num_gpus
        self.ctx = setting_ctx(self.num_gpus)
        self.num_workers = config.num_workers
        self.seq_len = 75
        self.build_model()

    def build_model(self):
        """
        Description : build network
        """
        #set network
        self.net = LipNet(self.dr_rate, self.batch_size, self.seq_len)
        self.net.hybridize()
        self.net.initialize(ctx=self.ctx)
        #set optimizer
        self.loss_fn = gluon.loss.CTCLoss()
        self.trainer = gluon.Trainer(self.net.collect_params(), \
                                     optimizer='SGD')

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
        input_transform = transforms.Compose([transforms.ToTensor(), \
                                             transforms.Normalize((0.7136, 0.4906, 0.3283), \
                                                                  (0.1138, 0.1078, 0.0917))])
        training_dataset = LipsDataset(self.image_path,
                                       self.align_path,
                                       transform=input_transform,
                                       seq_len=self.seq_len)

        train_dataloader = mx.gluon.data.DataLoader(training_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=self.num_workers)
        best_loss = sys.maxsize
        for epoch in trange(self.epochs):
            iter_no = 0
            for input_data, input_label in tqdm(train_dataloader):
                data = gluon.utils.split_and_load(input_data, self.ctx, even_split=False)
                label = gluon.utils.split_and_load(input_label, self.ctx, even_split=False)

                # pylint: disable=no-member
                sum_losses = 0
                len_losses = 0
                with autograd.record():
                    losses = [self.loss_fn(self.net(X), Y) for X, Y in zip(data, label)]
                for l in losses:
                    sum_losses += mx.nd.array(l).sum().asscalar()
                    len_losses += len(l)
                    l.backward()
                self.trainer.step(input_data.shape[0])
                if iter_no % 20 == 0:
                    current_loss = sum_losses / len_losses
                    print("epoch:{e} iter:{i} loss:{l}".format(e=epoch,
                                                               i=iter_no,
                                                               l=current_loss))
                    self.infer(data, label)
                    if best_loss > current_loss:
                        self.save_model(epoch, iter_no, current_loss)
                        best_loss = current_loss
                iter_no += 1

    def infer(self, input_data, input_label):
        """
        Description : Print sentence for prediction result
        """
        for data, label in zip(input_data, input_label):
            pred = self.net(data)
            pred_convert = char_beam_search(pred)
            label_convert = char_conv(label.asnumpy())
            for target, pred in zip(label_convert, pred_convert):
                print("target:{t}  pred:{p}".format(t=target, p=pred))
