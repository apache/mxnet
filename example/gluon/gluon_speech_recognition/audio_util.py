""" Code based on package python_speech_features

Author: James Lyons 2012
"""
import decimal

import numpy
import math


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'),
                                                rounding=decimal.ROUND_HALF_UP
                                                ))


def framesig(sig, frame_len, frame_step, winfunc=lambda x: numpy.ones((x,))):
    """Frame a signal into overlapping frames.
    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame
    that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no
    window is applied.
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((sig, zeros))

    indices = numpy.tile(
        numpy.arange(
            0, frame_len),
        (numframes, 1)) + numpy.tile(
            numpy.arange(
                0, numframes * frame_step, frame_step), (frame_len, 1)).T

    indices = numpy.array(indices, dtype=numpy.int32)
    frames = padsignal[indices]
    win = numpy.tile(winfunc(frame_len), (numframes, 1))
    return frames * win


def deframesig(frames, siglen, frame_len, frame_step,
               winfunc=lambda x: numpy.ones((x,))):
    """Does overlap-add procedure to undo the action of framesig.
    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output
    will be truncated to siglen samples.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame
    that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no
    window is applied.
    :returns: a 1-D signal.
    """
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = numpy.shape(frames)[0]
    assert numpy.shape(frames)[1] == frame_len, '"frames" matrix is wrong\
    size, 2nd dim is not equal to frame_len'

    indices = numpy.tile(
        numpy.arange(
            0, frame_len), (numframes, 1)) + numpy.tile(
                numpy.arange(
                    0, numframes * frame_step, frame_step), (frame_len, 1)).T

    indices = numpy.array(indices, dtype=numpy.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0:
        siglen = padlen

    rec_signal = numpy.zeros((padlen,))
    window_correction = numpy.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        # add a little bit so it is never zero
        window_correction[indices[i, :]] = window_correction[indices[i, :]] + \
                                           win + 1e-15
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]


def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an
    NxD matrix, output will be NxNFFT.
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are
    zero-padded.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will
    be the magnitude spectrum of the corresponding frame.
    """
    complex_spec = numpy.fft.rfft(frames, NFFT)
    return numpy.absolute(complex_spec)


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD
    matrix, output will be NxNFFT.
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are
    zero-padded.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will
    be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * numpy.square(magspec(frames, NFFT))


def logpowspec(frames, NFFT, norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an
    NxD matrix, output will be NxNFFT.
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are
    zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the
    max value (across all frames) is 1.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will
    be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames, NFFT)
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * numpy.log10(ps)
    if norm:
        return lps - numpy.max(lps)
    else:
        return lps


def preemphasis(signal, coeff=0.95):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return numpy.append(signal[0], signal[1:] - coeff * signal[:-1])


def delta(feat, N):
    """Compute delta features from a feature vector sequence.

    :param feat: A numpy array of size (NUMFRAMES by number of features)
    containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and
    following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features)
    containing delta features. Each row holds 1 delta feature vector.
    """
    NUMFRAMES = len(feat)
    feat = numpy.concatenate(([feat[0] for i in range(N)], feat, [feat[-1] for
                                                                  i in
                                                                  range(N)]))
    denom = sum([2 * i * i for i in range(1, N + 1)])
    dfeat = []
    for j in range(NUMFRAMES):
        dfeat.append(numpy.sum([n * feat[N + j + n]
                                for n in range(-1 * N, N + 1)], axis=0) /
                     denom)
    return dfeat
