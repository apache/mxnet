import numpy
import numpy as num
import stats
from common import *

class atrackReader(BaseReader):
    def __init__(self, featureFile, labelFile, byteOrder=None):
        BaseReader.__init__(self, featureFile, labelFile, byteOrder)

    def checkHeader(self, header):
        assert(header[0] == 0x56782)
        assert(header[1] == header[6]) # and header[1] == frameSize)
        assert(header[2] == header[5]) # and header[2] >= numSamples)
        assert(header[3] == 0)
        assert(header[4] == 24) # size of float + 20
        assert(header[4])

    def Read(self):
        # flip both the header and data using >
        # atrack format...
        """
        0.000000 354178 -2107177728
        0.000000 1845 889651200
        0.000000 1124588 -332918528
        0.000000 0 0
        0.000000 24 402653184
        0.000000 1124588 -332918528
        0.000000 1845 889651200
        -2.395848 -1072081519 -1856693824
        -1.677172 -1076449904 -1867655489
        -1.562828 -1077409088 -1073035073
        """
            
        f = open(self.featureFile, "rb")
        header = num.fromfile(f, dtype=num.dtype('>i4'), count=7)
        self.checkHeader(header)
        
        frameSize = header[1]
        numSamples = header[2]

        a = num.fromfile(f, dtype=num.dtype('>f4'), count=numSamples*frameSize)
        f.close()

        a = a.astype(num.float32)
        a = a.reshape((numSamples, frameSize))

        self._markDone()

        return a, ReadLabel(self.labelFile)