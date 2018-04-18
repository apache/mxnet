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
import mxnet as mx
import numpy as np
import gym
import cv2
import math
from threading import Thread
import time
import multiprocessing
import multiprocessing.pool
from flask import Flask, render_template, Response
import signal
import sys
is_py3 = sys.version[0] == '3'
if is_py3:
    import queue as queue
else:
    import Queue as queue

def make_web(queue):
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html')

    def gen():
        while True:
            frame = queue.get()
            _, frame = cv2.imencode('.JPEG', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame.tostring() + b'\r\n')

    @app.route('/video_feed')
    def video_feed():
        return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    try:
        app.run(host='0.0.0.0', port=8889)
    except:
        print('unable to open port')

def visual(X, show=True):
    X = X.transpose((0, 2, 3, 1))
    N = X.shape[0]
    n = int(math.ceil(math.sqrt(N)))
    h = X.shape[1]
    w = X.shape[2]
    buf = np.zeros((h*n, w*n, X.shape[3]), dtype=np.uint8)
    for i in range(N):
        x = i%n
        y = i//n
        buf[h*y:h*(y+1), w*x:w*(x+1), :] = X[i]
    if show:
        cv2.imshow('a', buf)
        cv2.waitKey(1)
    return buf

def env_step(args):
    return args[0].step(args[1])

class RLDataIter(object):
    def __init__(self, batch_size, input_length, nthreads=6, web_viz=False):
        super(RLDataIter, self).__init__()
        self.batch_size = batch_size
        self.input_length = input_length
        self.env = [self.make_env() for _ in range(batch_size)]
        self.act_dim = self.env[0].action_space.n

        self.state_ = None

        self.reset()

        self.provide_data = [mx.io.DataDesc('data', self.state_.shape, np.uint8)]

        self.web_viz = web_viz
        if web_viz:
            self.queue = queue.Queue()
            self.thread = Thread(target=make_web, args=(self.queue,))
            self.thread.daemon = True
            self.thread.start()

        self.nthreads = nthreads
        if nthreads > 1:
            self.pool = multiprocessing.pool.ThreadPool(6)

    def make_env(self):
        raise NotImplementedError()

    def reset(self):
        self.state_ = np.tile(
            np.asarray([env.reset() for env in self.env], dtype=np.uint8).transpose((0, 3, 1, 2)),
            (1, self.input_length, 1, 1))

    def visual(self):
        raise NotImplementedError()

    def act(self, action):
        if self.nthreads > 1:
            new = self.pool.map(env_step, zip(self.env, action))
        else:
            new = [env.step(act) for env, act in zip(self.env, action)]

        reward = np.asarray([i[1] for i in new], dtype=np.float32)
        done = np.asarray([i[2] for i in new], dtype=np.float32)

        channels = self.state_.shape[1]//self.input_length
        state = np.zeros_like(self.state_)
        state[:,:-channels,:,:] = self.state_[:,channels:,:,:]
        for i, (ob, env) in enumerate(zip(new, self.env)):
            if ob[2]:
                state[i,-channels:,:,:] = env.reset().transpose((2,0,1))
            else:
                state[i,-channels:,:,:] = ob[0].transpose((2,0,1))
        self.state_ = state

        if self.web_viz:
            try:
                while self.queue.qsize() > 10:
                    self.queue.get(False)
            except Empty:
                pass
            frame = self.visual()
            self.queue.put(frame)

        return reward, done

    def data(self):
        return [mx.nd.array(self.state_, dtype=np.uint8)]


class GymDataIter(RLDataIter):
    def __init__(self, game, batch_size, input_length, web_viz=False):
        self.game = game
        super(GymDataIter, self).__init__(batch_size, input_length, web_viz=web_viz)

    def make_env(self):
        return gym.make(self.game)

    def visual(self):
        data = self.state_[:4, -self.state_.shape[1]//self.input_length:, :, :]
        return visual(np.asarray(data, dtype=np.uint8), False)

if __name__ == '__main__':
    batch_size = 64
    dataiter = GymDataIter('Breakout-v0', batch_size, 4)
    dataiter.reset()
    tic = time.time()
    for _ in range(10):
        #data = dataiter.next().data[0].asnumpy().astype(np.uint8)
        #visual(data[:,-data.shape[1]/dataiter.input_length:,:,:])
        for _ in range(100):
            dataiter.act([env.action_space.sample() for env in dataiter.env])
            dataiter.clear_history()
            dataiter.next()
        print(batch_size*100/(time.time() - tic))
        tic = time.time()


