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
