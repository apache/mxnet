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
# pylint: disable=wildcard-import
"""Gluon EventHandlers for Estimators"""

__all__ = ['EventHandler', 'LoggingHandler']
import logging
import os
import time


class EventHandler(object):
    def __init__(self, estimator):
        self._estimator = estimator

    def train_begin(self):
        pass

    def train_end(self):
        pass

    def batch_begin(self):
        pass

    def batch_end(self):
        pass

    def epoch_begin(self):
        pass

    def epoch_end(self):
        pass


class LoggingHandler(EventHandler):
    """Basic Logging Handler that applies to every Gluon estimator by default.
    TODO: add doc
    """

    def __init__(self, estimator, log_name=None, file_name=None, file_location=None, ):
        super(LoggingHandler, self).__init__(estimator)
        log_name = log_name or 'Gluon Estimator'
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)
        streamhandler = logging.StreamHandler()
        self.logger.addHandler(streamhandler)
        # save logger to file only if file name or location is specified
        if file_name or file_location:
            file_name = file_name or log_name or 'estimator_log'
            file_location = file_location or './'
            filehandler = logging.FileHandler(os.path.join(file_location, file_name))
            self.logger.addHandler(filehandler)

    def train_begin(self):
        pass
        # logger.info(opt)

    def train_end(self):
        pass

    def batch_begin(self):
        self.batch_start = time.time()

    def batch_end(self):
        batch_time = time.time() - self.batch_start
        epoch = self._estimator.train_stats['epochs'][-1]
        step = self._estimator.train_stats['step']
        msg = '[Epoch %d] [Step %s] time/step: %.3fs ' % (epoch, step, batch_time)
        for key in self._estimator.train_stats.keys():
            if key.startswith('batch_'):
                msg += key[6:] + ': ' + '%.4f ' % self._estimator.train_stats[key]
        self.logger.info(msg)

    def epoch_begin(self):
        self.epoch_start = time.time()

    def epoch_end(self):
        epoch_time = time.time() - self.epoch_start
        epoch = self._estimator.train_stats['epochs'][-1]
        msg = 'Epoch %d finished in %.3fs: ' % (epoch, epoch_time)
        for key in self._estimator.train_stats.keys():
            if key.startswith('train_') or key.startswith('test_'):
                msg += key + ': ' + '%.4f ' % self._estimator.train_stats[key][epoch]
        self.logger.info(msg)