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
        self.image_path = config.image_path
        self.align_path = config.align_path
        self.num_gpus = config.num_gpus
        self.ctx = setting_ctx(self.num_gpus)
        self.num_workers = config.num_workers
        self.seq_len = 75

    def build_model(self, dr_rate=0, path=None):
        """
        Description : build network
        """
        #set network
        self.net = LipNet(dr_rate)
        self.net.hybridize()
        self.net.initialize(ctx=self.ctx)

        if path is not None:
            self.load_model(path)

        #set optimizer
        self.loss_fn = gluon.loss.CTCLoss()
        self.trainer = gluon.Trainer(self.net.collect_params(), \
                                     optimizer='SGD')

    def save_model(self, epoch, loss):
        """
        Description : save parameter of network weight
        """
        prefix = 'checkpoint/epoches'
        file_name = "{prefix}_{epoch}_loss_{l:.4f}".format(prefix=prefix,
                                                           epoch=str(epoch),
                                                           l=loss)
        self.net.save_parameters(file_name)

    def load_model(self, path=''):
        """
        Description : load parameter of network weight
        """
        self.net.load_parameters(path)

    def load_dataloader(self):
        """
        Description : Setup the dataloader
        """

        input_transform = transforms.Compose([transforms.ToTensor(), \
                                             transforms.Normalize((0.7136, 0.4906, 0.3283), \
                                                                  (0.1138, 0.1078, 0.0917))])
        training_dataset = LipsDataset(self.image_path,
                                       self.align_path,
                                       mode='train',
                                       transform=input_transform,
                                       seq_len=self.seq_len)

        self.train_dataloader = mx.gluon.data.DataLoader(training_dataset,
                                                         batch_size=self.batch_size,
                                                         shuffle=True,
                                                         num_workers=self.num_workers)

        valid_dataset = LipsDataset(self.image_path,
                                    self.align_path,
                                    mode='valid',
                                    transform=input_transform,
                                    seq_len=self.seq_len)

        self.valid_dataloader = mx.gluon.data.DataLoader(valid_dataset,
                                                         batch_size=self.batch_size,
                                                         shuffle=True,
                                                         num_workers=self.num_workers)

    def train(self, data, label, batch_size):
        """
        Description : training for LipNet
        """
        # pylint: disable=no-member
        sum_losses = 0
        len_losses = 0
        with autograd.record():
            losses = [self.loss_fn(self.net(X), Y) for X, Y in zip(data, label)]
        for loss in losses:
            sum_losses += mx.nd.array(loss).sum().asscalar()
            len_losses += len(loss)
            loss.backward()
        self.trainer.step(batch_size)
        return sum_losses, len_losses

    def infer(self, input_data, input_label):
        """
        Description : Print sentence for prediction result
        """
        sum_losses = 0
        len_losses = 0
        for data, label in zip(input_data, input_label):
            pred = self.net(data)
            sum_losses += mx.nd.array(self.loss_fn(pred, label)).sum().asscalar()
            len_losses += len(data)
            pred_convert = char_beam_search(pred)
            label_convert = char_conv(label.asnumpy())
            for target, pred in zip(label_convert, pred_convert):
                print("target:{t}  pred:{p}".format(t=target, p=pred))
        return sum_losses, len_losses

    def train_batch(self, dataloader):
        """
        Description : training for LipNet
        """
        sum_losses = 0
        len_losses = 0
        for input_data, input_label in tqdm(dataloader):
            data = gluon.utils.split_and_load(input_data, self.ctx, even_split=False)
            label = gluon.utils.split_and_load(input_label, self.ctx, even_split=False)
            batch_size = input_data.shape[0]
            sum_losses, len_losses = self.train(data, label, batch_size)
            sum_losses += sum_losses
            len_losses += len_losses

        return sum_losses, len_losses

    def infer_batch(self, dataloader):
        """
        Description : inference for LipNet
        """
        sum_losses = 0
        len_losses = 0
        for input_data, input_label in dataloader:
            data = gluon.utils.split_and_load(input_data, self.ctx, even_split=False)
            label = gluon.utils.split_and_load(input_label, self.ctx, even_split=False)
            sum_losses, len_losses = self.infer(data, label)
            sum_losses += sum_losses
            len_losses += len_losses

        return sum_losses, len_losses

    def run(self, epochs):
        """
        Description : Run training for LipNet
        """
        best_loss = sys.maxsize
        for epoch in trange(epochs):
            iter_no = 0

            ## train
            sum_losses, len_losses = self.train_batch(self.train_dataloader)

            if iter_no % 20 == 0:
                current_loss = sum_losses / len_losses
                print("[Train] epoch:{e} iter:{i} loss:{l:.4f}".format(e=epoch,
                                                                       i=iter_no,
                                                                       l=current_loss))

            ## validating
            sum_val_losses, len_val_losses = self.infer_batch(self.valid_dataloader)

            current_val_loss = sum_val_losses / len_val_losses
            print("[Vaild] epoch:{e} iter:{i} loss:{l:.4f}".format(e=epoch,
                                                                   i=iter_no,
                                                                   l=current_val_loss))

            if best_loss > current_val_loss:
                self.save_model(epoch, current_val_loss)
                best_loss = current_val_loss

            iter_no += 1
