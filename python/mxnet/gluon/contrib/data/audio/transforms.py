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

# coding: utf-8
# pylint: disable= arguments-differ
"Audio transforms."

import warnings
import numpy as np
try:
    import librosa
except ImportError as e:
    warnings.warn("gluon/contrib/data/audio/transforms.py : librosa dependency could not be resolved or \
    imported, could not provide some/all transform.")

from ..... import ndarray as nd
from ....block import Block


class Loader(Block):
    """
        This transform opens a filepath and converts that into an NDArray using librosa to load
        res_type='kaiser_fast'

    Parameters
    ----------
    Keyword arguments that can be passed, which are utilized by librosa module are:
    res_type: string, default 'kaiser_fast' the resampling type - other accepateble values: 'kaiser_best'.


    Inputs:
        - **x**: audio file path to be loaded on to the NDArray.

    Outputs:
        - **out**: output array is a scaled NDArray with (samples, ).

    """

    def __init__(self, res_type='kaiser_fast'):
        super(Loader, self).__init__()
        self._restype = res_type

    def forward(self, x):
        if not librosa:
            raise RuntimeError("Librosa dependency is not installed! Install that and retry!")
        X1, _ = librosa.load(x, res_type=self._restype)
        return nd.array(X1)


class MFCC(Block):
    """
        Extracts Mel frequency cepstrum coefficients from the audio data file
        More details : https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html

    Parameters
    ----------
    Keyword arguments that can be passed, which are utilized by librosa module are:
    sr: int, default 22050
        sampling rate of the input audio signal
    n_mfcc: int, default 20
        number of mfccs to return


    Inputs:
        - **x**: input tensor (samples, ) shape.

    Outputs:
        - **out**: output array is a scaled NDArray with (samples, ) shape.

    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super(MFCC, self).__init__()

    def forward(self, x):
        if not librosa:
            warnings.warn("Librosa dependency is not installed! Install that and retry")
            return x
        if isinstance(x, np.ndarray):
            y = x
        elif isinstance(x, nd.NDArray):
            y = x.asnumpy()
        else:
            warnings.warn("Input object is not numpy or NDArray... Cannot apply the transform: MFCC!")
            return x

        audio_tmp = np.mean(librosa.feature.mfcc(y=y, **self.kwargs).T, axis=0)
        return nd.array(audio_tmp)


class Scale(Block):
    """Scale audio numpy.ndarray from a 16-bit integer to a floating point number between
    -1.0 and 1.0. The 16-bit integer is the sample resolution or bit depth.

    Parameters
    ----------
    scale_factor : float
        The factor to scale the input tensor by.


    Inputs:
        - **x**: input tensor (samples, ) shape.

    Outputs:
        - **out**: output array is a scaled NDArray with (samples, ) shape.

    Examples
    --------
    >>> scale = audio.transforms.Scale(scale_factor=2)
    >>> audio_samples = mx.nd.array([2,3,4])
    >>> scale(audio_samples)
    [1.  1.5 2. ]
    <NDArray 3 @cpu(0)>

    """

    def __init__(self, scale_factor=2**31):
        self.scale_factor = scale_factor
        super(Scale, self).__init__()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            return nd.array(x/self.scale_factor)
        return x / self.scale_factor


class PadTrim(Block):
    """Pad/Trim a 1d-NDArray of NPArray (Signal or Labels)

    Parameters
    ----------
    max_len : int
        Length to which the array will be padded or trimmed to.
    fill_value: int or float
        If there is a need of padding, what value to padd at the end of the input array


    Inputs:
        - **x**: input tensor (samples, ) shape.

    Outputs:
        - **out**: output array is a scaled NDArray with (max_len, ) shape.

    Examples
    --------
    >>> padtrim = audio.transforms.PadTrim(max_len=9, fill_value=0)
    >>> audio_samples = mx.nd.array([1,2,3,4,5])
    >>> padtrim(audio_samples)
    [1. 2. 3. 4. 5. 0. 0. 0. 0.]
    <NDArray 9 @cpu(0)>

    """

    def __init__(self, max_len, fill_value=0):
        self._max_len = max_len
        self._fill_value = fill_value
        super(PadTrim, self).__init__()

    def forward(self, x):
        if  isinstance(x, np.ndarray):
            x = nd.array(x)
        if self._max_len > x.size:
            pad = nd.ones((self._max_len - x.size,)) * self._fill_value
            x = nd.concat(x, pad, dim=0)
        elif self._max_len < x.size:
            x = x[:self._max_len]
        return x


class MEL(Block):
    """Create MEL Spectrograms from a raw audio signal. Relatively pretty slow.

    Parameters
    ----------
    Keyword arguments that can be passed, which are utilized by librosa module are:
    sr: int, default 22050
        sampling rate of the input audio signal
    n_fft: int, default 2048
        length of the Fast fourier transform window
    n_mels: int,
        number of mel bands to generate
    hop_length: int, default 512
        total samples between successive frames


    Inputs:
        - **x**: input tensor (samples, ) shape.

    Outputs:
        - **out**: output array which consists of mel spectograms, shape = (n_mels, 1)

       Usage (see librosa.feature.melspectrogram docs):
           MEL(sr=16000, n_fft=1600, hop_length=800, n_mels=64)

    Examples
    --------
    >>> mel = audio.transforms.MEL()
    >>> audio_samples = mx.nd.array([1,2,3,4,5])
    >>> mel(audio_samples)
    [[3.81801406e+04]
    [9.86858240e-29]
    [1.87405472e-29]
    [2.38637225e-29]
    [3.94043010e-29]
    [3.67071565e-29]
    [7.29390295e-29]
    [8.84324438e-30]...
    <NDArray 128x1 @cpu(0)>

    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super(MEL, self).__init__()

    def forward(self, x):
        if librosa is None:
            warnings.warn("Cannot create spectrograms, since dependency librosa is not installed!")
            return x
        if isinstance(x, nd.NDArray):
            x = x.asnumpy()
        specs = librosa.feature.melspectrogram(x, **self.kwargs)
        return nd.array(specs)
    