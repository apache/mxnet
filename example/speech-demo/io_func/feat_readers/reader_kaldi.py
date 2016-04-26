from common import *

import random
import time

import ctypes
import numpy
import sys
import re

c_float_ptr = ctypes.POINTER(ctypes.c_float)
c_int_ptr = ctypes.POINTER(ctypes.c_int)
c_void_p = ctypes.c_void_p
c_int = ctypes.c_int
c_char_p = ctypes.c_char_p
c_float = ctypes.c_float

kaldi = ctypes.cdll.LoadLibrary("libkaldi-python-wrap.so")  # this needs to be in LD_LIBRARY_PATH

def decl(f, restype, argtypes):
    f.restype = restype
    if argtypes is not None and len(argtypes) != 0:
        f.argtypes = argtypes

decl(kaldi.SBFMReader_new,          c_void_p,   [])
decl(kaldi.SBFMReader_new_char,     c_void_p,   [c_char_p])
decl(kaldi.SBFMReader_Open,         c_int,      [c_void_p, c_char_p])
decl(kaldi.SBFMReader_Done,         c_int,      [c_void_p])
decl(kaldi.SBFMReader_Key,          c_char_p,   [c_void_p])
decl(kaldi.SBFMReader_FreeCurrent,  None,       [c_void_p])
decl(kaldi.SBFMReader_Value,        c_void_p,   [c_void_p])
decl(kaldi.SBFMReader_Next,         None,       [c_void_p])
decl(kaldi.SBFMReader_IsOpen,       c_int,      [c_void_p])
decl(kaldi.SBFMReader_Close,        c_int,      [c_void_p])
decl(kaldi.SBFMReader_Delete,       None,       [c_void_p])

decl(kaldi.MatrixF_NumRows,     c_int,       [c_void_p])
decl(kaldi.MatrixF_NumCols,     c_int,       [c_void_p])
decl(kaldi.MatrixF_Stride,      c_int,       [c_void_p])
decl(kaldi.MatrixF_cpy_to_ptr,  None,        [c_void_p, c_float_ptr, c_int])
decl(kaldi.MatrixF_SizeInBytes, c_int,       [c_void_p])
decl(kaldi.MatrixF_Data,        c_float_ptr, [c_void_p])

decl(kaldi.RAPReader_new_char,      c_void_p,   [c_char_p])
decl(kaldi.RAPReader_HasKey,        c_int,      [c_void_p, c_char_p])
decl(kaldi.RAPReader_Value,         c_int_ptr,  [c_void_p, c_char_p])
decl(kaldi.RAPReader_DeleteValue,   None,       [c_void_p, c_int_ptr])
decl(kaldi.RAPReader_Delete,        None,       [c_void_p])

decl(kaldi.Nnet_new,            c_void_p,   [c_char_p, c_float, c_int])
decl(kaldi.Nnet_Feedforward,    c_void_p,   [c_void_p, c_void_p])
decl(kaldi.Nnet_Delete,         None,       [c_void_p])

class kaldiReader(BaseReader):
    def __init__(self, featureFile, labelFile, byteOrder=None):
        BaseReader.__init__(self, featureFile, labelFile, byteOrder)

        arr = re.split('\s+', featureFile, maxsplit=1)
        if len(arr) != 2:
            raise Exception("two items required in featureFile line: <transform> <rspecifier>")
        feature_transform, featureFile = arr
        if feature_transform == "NO_FEATURE_TRANSFORM":
            feature_transform = None

        self.feature_rspecifier = featureFile
        self.targets_rspecifier = labelFile
        self.feature_reader = kaldi.SBFMReader_new_char(self.feature_rspecifier)

        if self.targets_rspecifier is not None:
            self.targets_reader = kaldi.RAPReader_new_char(self.targets_rspecifier)
        if feature_transform is not None:
            self.nnet_transf = kaldi.Nnet_new(feature_transform, ctypes.c_float(1.0), 1)
        else:
            self.nnet_transf = None

    def Cleanup(self):
        kaldi.SBFMReader_Delete(self.feature_reader)
        if self.targets_rspecifier is not None:
            kaldi.RAPReader_Delete(self.targets_reader)
        if self.nnet_transf is not None:
            kaldi.Nnet_Delete(self.nnet_transf)

    def Read(self):
        if kaldi.SBFMReader_Done(self.feature_reader):
            self._markDone()
            return None
        utt = kaldi.SBFMReader_Key(self.feature_reader)
        self.utt_id = utt

        #return numpy.ones((256, 819)).astype('float32'), numpy.ones(256).astype('int32')

        feat_value = kaldi.SBFMReader_Value(self.feature_reader)
        if self.nnet_transf is not None:
            feat_value = kaldi.Nnet_Feedforward(self.nnet_transf, feat_value)
        feat_rows = kaldi.MatrixF_NumRows(feat_value)
        feat_cols = kaldi.MatrixF_NumCols(feat_value)
        feat_data = kaldi.MatrixF_Data(feat_value)
        
        # never use numpy.ndarray(buf=) or numpy.ctypeslib.as_array
        # because you don't know if Python or C owns buffer
        # (even if you numpy.copy() resulting array)
        # http://stackoverflow.com/questions/4355524/getting-data-from-ctypes-array-into-numpy
        #
        # Can't use memmove/memcpy because arrays are strided
        # Use special function -_-

        feats = numpy.empty((feat_rows,feat_cols), dtype=numpy.float32)
        # MUST: cast Python int to pointer, otherwise C interprets as 32-bit
        # if you print the pointer value before casting, you might see weird value before seg fault
        # casting fixes that
        feats_numpy_ptr = ctypes.cast(feats.ctypes.data, c_float_ptr)
        kaldi.MatrixF_cpy_to_ptr(feat_value, feats_numpy_ptr, feats.strides[0]/4)

        if self.targets_rspecifier is not None:
            if kaldi.RAPReader_HasKey(self.targets_reader, utt):
                tgt_value = kaldi.RAPReader_Value(self.targets_reader, utt)
                
                tgts = numpy.empty((feat_rows,), dtype=numpy.int32)
                # ok to use memmove because this is 1-dimensional array I made in C (no stride)
                tgts_numpy_ptr = ctypes.cast(tgts.ctypes.data, c_int_ptr)
                ctypes.memmove(tgts_numpy_ptr, tgt_value, 4 * feat_rows)

                kaldi.RAPReader_DeleteValue(self.targets_reader, tgt_value)
            else:
                tgts = None
        else:
            tgts = None
        
        kaldi.SBFMReader_Next(self.feature_reader)

        #print "FEATS:", feats[0:5][0:5]
        #print "TGTS :", tgts[0:5]

        return feats, tgts

    def GetUttId(self):
        return self.utt_id
