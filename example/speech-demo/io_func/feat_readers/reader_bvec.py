from __future__ import print_function
import struct
import array
import numpy
from common import *

class bvecReader(BaseReader):

    def __init__(self, featureFile, labelFile, byteOrder=None):
        BaseReader.__init__(self, featureFile, labelFile, byteOrder)

    def Read(self):

        with open(self.featureFile,"rb") as f:

            dt = numpy.dtype([('numSamples',(numpy.int32,1)),('dim',(numpy.int32,1))])
            header =  numpy.fromfile(f,dt.newbyteorder('>'),count=1)

            numSamples = header[0]['numSamples']
            dim        = header[0]['dim']

            print('Num samples = {}'.format(numSamples))
            print('dim = {}'.format(dim))

            dt = numpy.dtype([('sample',(numpy.float32,dim))]) 
            samples = numpy.fromfile(f,dt.newbyteorder('>'),count=numSamples)

        self._markDone()

        return samples[:]['sample'], ReadLabel(self.labelFile)
