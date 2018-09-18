''' Code partially copied from python_speech_features package
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import audio_util as sigproc

import os
import numpy as np
import logging

from scipy import signal
from scipy.fftpack import dct
import librosa


class Feature(object):
    """ Base class for features calculation
    All children class must implement __str__ and _call function.

    # Arguments
        fs: sampling frequency of audio signal. If the audio has not this fs,
        it will be resampled
        eps
    """

    def __init__(self, fs=16e3, eps=1e-8, stride=1, num_context=0,
                 mean_norm=True, var_norm=True):
        self.fs = fs
        self.eps = eps

        self.mean_norm = mean_norm
        self.var_norm = var_norm

        self.stride = stride
        self.num_context = num_context
        self._logger = logging.getLogger('%s.%s' % (__name__,
                                                    self.__class__.__name__))

    def __call__(self, audio):
        """ This method load the audio and do the transformation of signal

        # Inputs
            audio:
                if audio is a string and the file exists, the wave file will
                be loaded and resampled (if necessary) to fs
                if audio is a ndarray or list and is not empty, it will make
                the transformation without any resampling

        # Exception
            TypeError if audio were not recognized

        """
        if ((isinstance(audio, str) or isinstance(audio, unicode))
            and os.path.isfile(audio)):
            audio, current_fs = librosa.audio.load(audio)
            audio = librosa.core.resample(audio, current_fs, self.fs)
            feats = self._call(audio)
        elif type(audio) in (np.ndarray, list) and len(audio) > 1:
            feats = self._call(audio)
        else:
            TypeError("audio type is not support")

        return self._standarize(self._postprocessing(feats))

    def _call(self, data):
        raise NotImplementedError("__call__ must be overrided")

    def _standarize(self, feats):
        if self.mean_norm:
            feats -= np.mean(feats, axis=0, keepdims=True)
        if self.var_norm:
            feats /= (np.std(feats, axis=0, keepdims=True) + self.eps)
        return feats

    def _postprocessing(self, feats):
        # Code adapted from
        # https://github.com/mozilla/DeepSpeech/blob/master/util/audio.py

        # We only keep every second feature (BiRNN stride = 2)
        feats = feats[::self.stride]

        if self.num_context == 0:
            return feats
        num_feats = feats.shape[1]

        train_inputs = np.array([], np.float32)
        train_inputs.resize((feats.shape[0],
                            num_feats + 2*num_feats*self.num_context))

        # Prepare pre-fix post fix context
        # (TODO: Fill empty_mfcc with MCFF of silence)
        empty_mfcc = np.array([])
        empty_mfcc.resize((num_feats))

        # Prepare train_inputs with past and future contexts
        time_slices = range(train_inputs.shape[0])
        context_past_min = time_slices[0] + self.num_context
        context_future_max = time_slices[-1] - self.num_context
        for time_slice in time_slices:
            # Reminder: array[start:stop:step]
            # slices from indice |start| up to |stop| (not included), every
            # |step|
            # Pick up to self.num_context time slices in the past, and complete
            # with empty
            # mfcc features
            need_empty_past = max(0, (context_past_min - time_slice))
            empty_source_past = list(empty_mfcc for empty_slots
                                     in range(need_empty_past))
            data_source_past = feats[max(0, time_slice -
                                         self.num_context):time_slice]
            assert(len(empty_source_past) +
                   len(data_source_past) == self.num_context)

            # Pick up to self.num_context time slices in the future, and
            # complete with empty
            # mfcc features
            need_empty_future = max(0, (time_slice - context_future_max))
            empty_source_future = list(empty_mfcc
                                       for empty_slots in
                                       range(need_empty_future))
            data_source_future = feats[time_slice + 1:time_slice +
                                       self.num_context + 1]

            assert(len(empty_source_future) +
                   len(data_source_future) == self.num_context)

            if need_empty_past:
                past = np.concatenate((empty_source_past, data_source_past))
            else:
                past = data_source_past

            if need_empty_future:
                future = np.concatenate((data_source_future,
                                         empty_source_future))
            else:
                future = data_source_future

            past = np.reshape(past, self.num_context*num_feats)
            now = feats[time_slice]
            future = np.reshape(future, self.num_context*num_feats)

            train_inputs[time_slice] = np.concatenate((past, now, future))
            assert(len(train_inputs[time_slice])
                   == num_feats + 2*num_feats*self.num_context)

        self._num_feats = num_feats + 2*num_feats*self.num_context

        return train_inputs

    def __str__(self):
        raise NotImplementedError("__str__ must be overrided")

    @property
    def num_feats(self):
        return self._num_feats


class FBank(Feature):
    """Compute Mel-filterbank energy features from an audio signal.

    # Arguments
        win_len: the length of the analysis window in seconds.
            Default  is 0.025s (25 milliseconds)
        win_step: the step between successive windows in seconds.
            Default is 0.01s (10 milliseconds)
        num_filt: the number of filters in the filterbank, default 40.
        nfft: the FFT size. Default is 512.
        low_freq: lowest band edge of mel filters in Hz.
            Default is 20.
        high_freq: highest band edge of mel filters in Hz.
            Default is 7800
        pre_emph: apply preemphasis filter with preemph as coefficient.
        0 is no filter. Default is 0.97.
        win_func: the analysis window to apply to each frame.
            By default hamming window is applied.
    """

    def __init__(self, win_len=0.025, win_step=0.01,
                 num_filt=40, nfft=512, low_freq=20, high_freq=7800,
                 pre_emph=0.97, win_fun=signal.hamming, **kwargs):

        super(FBank, self).__init__(**kwargs)

        if high_freq > self.fs / 2:
            raise ValueError("high_freq must be less or equal than fs/2")

        self.win_len = win_len
        self.win_step = win_step
        self.num_filt = num_filt
        self.nfft = nfft
        self.low_freq = low_freq
        self.high_freq = high_freq or self.fs / 2
        self.pre_emph = pre_emph
        self.win_fun = win_fun
        self._filterbanks = self._get_filterbanks()

        self._num_feats = self.num_filt

    @property
    def mel_points(self):
        return np.linspace(self._low_mel, self._high_mel, self.num_filt + 2)

    @property
    def low_freq(self):
        return self._low_freq

    @low_freq.setter
    def low_freq(self, value):
        self._low_mel = self._hz2mel(value)
        self._low_freq = value

    @property
    def high_freq(self):
        return self._high_freq

    @high_freq.setter
    def high_freq(self, value):
        self._high_mel = self._hz2mel(value)
        self._high_freq = value

    def _call(self, signal):
        """Compute Mel-filterbank energy features from an audio signal.
        :param signal: the audio signal from which to compute features. Should
        be an N*1 array

        Returns:
            2 values. The first is a numpy array of size (NUMFRAMES by nfilt)
            containing features. Each row holds 1 feature vector. The
            second return value is the energy in each frame (total energy,
            unwindowed)
        """

        signal = sigproc.preemphasis(signal, self.pre_emph)

        frames = sigproc.framesig(signal,
                                  self.win_len * self.fs,
                                  self.win_step * self.fs,
                                  self.win_fun)

        pspec = sigproc.powspec(frames, self.nfft)
        # this stores the total energy in each frame
        energy = np.sum(pspec, 1)
        # if energy is zero, we get problems with log
        energy = np.where(energy == 0, np.finfo(float).eps, energy)

        # compute the filterbank energies
        feat = np.dot(pspec, self._filterbanks.T)
        # if feat is zero, we get problems with log
        feat = np.where(feat == 0, np.finfo(float).eps, feat)

        return feat, energy

    def _get_filterbanks(self):
        """Compute a Mel-filterbank. The filters are stored in the rows, the
        columns correspond
        to fft bins. The filters are returned as an array of size nfilt *
        (nfft / 2 + 1)

        Returns:
            A numpy array of size num_filt * (nfft/2 + 1) containing
            filterbank. Each row holds 1 filter.
        """

        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = np.floor((self.nfft + 1) * self._mel2hz(self.mel_points) /
                       self.fs)

        fbank = np.zeros([self.num_filt, int(self.nfft / 2 + 1)])
        for j in range(0, self.num_filt):
            for i in range(int(bin[j]), int(bin[j + 1])):
                fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
            for i in range(int(bin[j + 1]), int(bin[j + 2])):
                fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
        return fbank

    def _hz2mel(self, hz):
        """Convert a value in Hertz to Mels

        Args:
            hz: a value in Hz. This can also be a numpy array, conversion
            proceeds element-wise.

        Returns:
            A value in Mels. If an array was passed in, an identical sized
            array is returned.
        """
        return 2595 * np.log10(1 + hz / 700.0)

    def _mel2hz(self, mel):
        """Convert a value in Mels to Hertz

        Args:
            mel: a value in Mels. This can also be a numpy array, conversion
            proceeds element-wise.

        Returns:
            A value in Hertz. If an array was passed in, an identical sized
            array is returned.
        """
        return 700 * (10**(mel / 2595.0) - 1)

    def __str__(self):
        return "fbank"


class MFCC(FBank):
    """Compute MFCC features from an audio signal.

    # Arguments
        num_cep: the number of cepstrum to return. Default 13.
        cep_lifter: apply a lifter to final cepstral coefficients. 0 is
        no lifter. Default is 22.
        append_energy: if this is true, the zeroth cepstral coefficient
        is replaced with the log of the total frame energy.
        d: if True add deltas coeficients. Default True
        dd: if True add delta-deltas coeficients. Default True
        norm: if 'cmn' performs the cepstral mean normalization. elif 'cmvn'
        performs the cepstral mean and variance normalizastion. Default 'cmn'
    """

    def __init__(self, num_cep=13, cep_lifter=22, append_energy=True,
                 d=True, dd=True, **kwargs):

        super(MFCC, self).__init__(**kwargs)

        self.num_cep = num_cep
        self.cep_lifter = cep_lifter
        self.append_energy = append_energy
        self.d = d
        self.dd = dd
        self._num_feats = (1 + self.d + self.dd) * self.num_cep

        self._logger = logging.getLogger('%s.%s' % (__name__,
                                                    self.__class__.__name__))

    def _call(self, signal):
        """Compute MFCC features from an audio signal.

        Args:
            signal: the audio signal from which to compute features. Should be
            an N*1 array

        Returns:
            A numpy array of size (NUMFRAMES by numcep) containing features.
            Each row holds 1 feature vector.
        """
        feat, energy = super(MFCC, self)._call(signal)

        feat = np.log(feat)
        feat = dct(feat, type=2, axis=1, norm='ortho')[:, :self.num_cep]
        feat = self._lifter(feat, self.cep_lifter)

        if self.append_energy:
            # replace first cepstral coefficient with log of frame energy
            feat[:, 0] = np.log(energy + self.eps)

        if self.d:
            d = sigproc.delta(feat, 2)
            feat = np.hstack([feat, d])

            if self.dd:
                feat = np.hstack([feat, sigproc.delta(d, 2)])

        return feat

    def _lifter(self, cepstra, L=22):
        """Apply a cepstral lifter the the matrix of cepstra.

        This has the effect of increasing the magnitude of the high frequency
        DCT coeffs.

        Args:
            cepstra: the matrix of mel-cepstra, will be numframes * numcep in
            size.
            L: the liftering coefficient to use. Default is 22. L <= 0 disables
            lifter.
        """
        if L > 0:
            nframes, ncoeff = np.shape(cepstra)
            n = np.arange(ncoeff)
            lift = 1 + (L / 2) * np.sin(np.pi * n / L)
            return lift * cepstra
        else:
            # values of L <= 0, do nothing
            return cepstra

    def __str__(self):
        return "mfcc"


class LogFbank(FBank):
    """Compute Mel-filterbank energy features from an audio signal.

    # Arguments
        append_energy: if this is true, log of the total frame energy is
        append to the features vector. Default False
        d: if True add deltas coeficients. Default False
        dd: if True add delta-deltas coeficients. Default False
    """

    def __init__(self, d=False, dd=False, append_energy=False, **kwargs):
        """Constructor
        """

        super(LogFbank, self).__init__(**kwargs)

        self.d = d
        self.dd = dd
        self.append_energy = append_energy
        self._num_feats = ((1 + self.d + self.dd)
                           * (self.num_filt + self.append_energy))

        self._logger = logging.getLogger('%s.%s' % (__name__,
                                                    self.__class__.__name__))

    def _call(self, signal):
        """Compute log Mel-filterbank energy features from an audio signal.
        :param signal: the audio signal from which to compute features. Should
        be an N*1 array

        Returns:
             A numpy array of size (NUMFRAMES by nfilt) containing features.
             Each row holds 1 feature vector.
        """
        feat, energy = super(LogFbank, self)._call(signal)

        feat = np.log(feat)

        if self.append_energy:
            feat = np.hstack([feat, np.log(energy + self.eps)[:, np.newaxis]])

        if self.d:
            d = sigproc.delta(feat, 2)
            feat = np.hstack([feat, d])

            if self.dd:
                feat = np.hstack([feat, sigproc.delta(d, 2)])

        return feat

    def __str__(self):
        return "logfbank"


class Raw(Feature):
    """ Raw features extractor
    """
    def __init__(self, **kwargs):
        super(Raw, self).__init__(**kwargs)
        self._num_feats = None

    def _call(self, x):
        return x

    def _postprocessing(self, x):
        return x

    def __str__(self):
        return "raw"


raw = Raw()
