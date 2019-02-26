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
Description : LipNet module using gluon
"""

from mxnet.gluon import nn, rnn
# pylint: disable=too-many-instance-attributes
class LipNet(nn.HybridBlock):
    """
    Description : LipNet network using gluon
    dr_rate : Dropout rate
    """
    def __init__(self, dr_rate, **kwargs):
        super(LipNet, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv3D(32, kernel_size=(3, 5, 5), strides=(1, 2, 2), padding=(1, 2, 2))
            self.bn1 = nn.InstanceNorm(in_channels=32)
            self.dr1 = nn.Dropout(dr_rate, axes=(1, 2))
            self.pool1 = nn.MaxPool3D((1, 2, 2), (1, 2, 2))
            self.conv2 = nn.Conv3D(64, kernel_size=(3, 5, 5), strides=(1, 1, 1), padding=(1, 2, 2))
            self.bn2 = nn.InstanceNorm(in_channels=64)
            self.dr2 = nn.Dropout(dr_rate, axes=(1, 2))
            self.pool2 = nn.MaxPool3D((1, 2, 2), (1, 2, 2))
            self.conv3 = nn.Conv3D(96, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding=(1, 2, 2))
            self.bn3 = nn.InstanceNorm(in_channels=96)
            self.dr3 = nn.Dropout(dr_rate, axes=(1, 2))
            self.pool3 = nn.MaxPool3D((1, 2, 2), (1, 2, 2))
            self.gru1 = rnn.GRU(256, bidirectional=True)
            self.gru2 = rnn.GRU(256, bidirectional=True)
            self.dense = nn.Dense(27+1, flatten=False)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dr1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dr2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.dr3(out)
        out = self.pool3(out)
        out = F.transpose(out, (2, 0, 1, 3, 4))
        # pylint: disable=no-member
        out = out.reshape((0, 0, -1))
        out = self.gru1(out)
        out = self.gru2(out)
        out = self.dense(out)
        out = F.log_softmax(out, axis=2)
        out = F.transpose(out, (1, 0, 2))
        return out
