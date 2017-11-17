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
