import numpy
import stats
from common import *

class htkReader(BaseReader):
    def __init__(self, featureFile, labelFile, byteOrder=None):
        BaseReader.__init__(self, featureFile, labelFile, byteOrder)

    def Read(self):

        #return numpy.ones((256, 819)).astype('float32'), numpy.ones(256).astype('int32')

        with open(self.featureFile,"rb") as f:

            dt = numpy.dtype([('numSamples',(numpy.int32,1)),('sampPeriod',(numpy.int32,1)),('sampSize',(numpy.int16,1)),('sampKind',(numpy.int16,1))])
            header =  numpy.fromfile(f,dt.newbyteorder('>' if self.byteOrder==ByteOrder.BigEndian else '<'),count=1)

            numSamples = header[0]['numSamples']
            sampPeriod = header[0]['sampPeriod']
            sampSize   = header[0]['sampSize']
            sampKind   = header[0]['sampKind']

            # print 'Num samples = {}'.format(numSamples)
            # print 'Sample period = {}'.format(sampPeriod)
            # print 'Sample size = {}'.format(sampSize)
            # print 'Sample kind = {}'.format(sampKind)
            dt = numpy.dtype([('sample',(numpy.float32,sampSize/4))]) 
            samples = numpy.fromfile(f,dt.newbyteorder('>' if self.byteOrder==ByteOrder.BigEndian else '<'),count=numSamples)

        self._markDone()

        if self.labelFile is None:
            labels = None
        else:
            labels = ReadLabel(self.labelFile)
            
        return samples[:]['sample'], labels
