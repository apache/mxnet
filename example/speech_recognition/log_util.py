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

import logging
import logging.handlers
from singleton import Singleton

@Singleton
class LogUtil:

    _logger = None
    _filename = None

    def getlogger(self, filename=None):
        if self._logger is not None and filename is not None:
            self._logger.warning('Filename %s ignored, logger is already instanciated with %s' % (filename, self._filename))
        if self._logger is None:
            self._filename = filename

            # logger
            self._logger = logging.getLogger('logger')
            # remove default handler
            self._logger.propagate = False

            stream_handler = logging.StreamHandler()
            stream_formatter = logging.Formatter('[%(levelname)8s][%(asctime)s.%(msecs)03d] %(message)s',
                                                 datefmt='%Y/%m/%d %H:%M:%S')
            stream_handler.setFormatter(stream_formatter)

            if self._filename is not None:
                file_max_bytes = 10 * 1024 * 1024

                file_handler = logging.handlers.RotatingFileHandler(filename='./log/' + self._filename,
                                                                   maxBytes=file_max_bytes,
                                                                   backupCount=10)
                file_formatter = logging.Formatter('[%(levelname)8s][%(asctime)s.%(msecs)03d] %(message)s',
                                                   datefmt='%Y/%m/%d %H:%M:%S')
                file_handler.setFormatter(file_formatter)
                self._logger.addHandler(file_handler)

            self._logger.addHandler(stream_handler)
            self._logger.setLevel(logging.DEBUG)
            
        
        return self._logger

