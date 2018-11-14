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

"""Testing audio transforms in gluon container."""
from __future__ import print_function
import numpy as np
from mxnet import gluon
from mxnet.gluon.contrib.data.audio import transforms
from mxnet.test_utils import assert_almost_equal
from common import with_seed


@with_seed()
def test_pad_trim():
    """
        Function to test Pad/Trim Audio transform
    """
    data_in = np.random.randint(1, high=20, size=(15))
    # trying trimming the audio samples here...
    max_len = 10
    pad_trim = gluon.contrib.data.audio.transforms.PadTrim(max_len=max_len)
    trimmed_audio = pad_trim(data_in)
    np_trimmed = data_in[:max_len]
    assert_almost_equal(trimmed_audio.asnumpy(), np_trimmed)

    #trying padding here...
    max_len = 25
    fill_value = 0
    pad_trim = transforms.PadTrim(max_len=max_len, fill_value=fill_value)
    np_padded = np.pad(data_in, pad_width=max_len-len(data_in), mode='constant', \
                        constant_values=fill_value)[max_len-len(data_in):]
    padded_audio = pad_trim(data_in)
    assert_almost_equal(padded_audio.asnumpy(), np_padded)


@with_seed()
def test_scale():
    """
        Function to test scaling of the audio transform
    """
    data_in = np.random.randint(1, high=20, size=(15))
    # Scaling the audio signal meaning dividing each sample by the scaling factor
    scale_factor = 2.0
    scaled_numpy = data_in /scale_factor
    scale = transforms.Scale(scale_factor=scale_factor)
    scaled_audio = scale(data_in)
    assert_almost_equal(scaled_audio.asnumpy(), scaled_numpy)


@with_seed()
def test_mfcc():
    """
        Function to test extraction of mfcc from audio signal
    """
    audio_samples = np.random.rand(20)
    n_mfcc = 64
    mfcc = gluon.contrib.data.audio.transforms.MFCC(n_mfcc=n_mfcc)

    mfcc_features = mfcc(audio_samples)
    assert mfcc_features.shape[0] == n_mfcc


@with_seed()
def test_mel():
    """
        Function to test extraction of MEL spectrograms from audio signal
    """
    audio_samples = np.random.rand(20)
    n_mels = 256
    mel = gluon.contrib.data.audio.transforms.MEL(n_mels=n_mels)

    mels = mel(audio_samples)
    assert mels.shape[0] == n_mels

if __name__ == '__main__':
    import nose
    nose.runmodule()
