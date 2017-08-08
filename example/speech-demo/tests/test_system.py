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
from pdnn.run_DNN import run_DNN
from pdnn.run_RBM import run_RBM
from pdnn.run_SDA import run_SDA
from pdnn.eval_DNN import eval_DNN
import json
from utils.utils import setup_logger

MNIST_CONF = json.load(open("configs/unittest_mnist.json"))
MAX_ITERS = 2
setup_logger(None)

def banner(s):
    print("***********************" + s + "*************************")

def test_hi():
    print("hi")

def test_rbm_dnn():
    banner("rbm dnn")
    mnist_conf = MNIST_CONF.copy()

    mnist_conf["train_rbm"]["max_iters"] = MAX_ITERS
    run_RBM(mnist_conf)

    mnist_conf["train_dnn"]["max_iters"] = MAX_ITERS
    mnist_conf["init_dnn"] = {
        "filename": "temp/rbm/final.nnet",
        "num_hidden_layers": -1,
        "with_final": 1
    }
    run_DNN(mnist_conf)

    mnist_conf["init_rbm"] = {
        "filename": "temp/dnn/final.nnet",
        "num_hidden_layers": -1,
        "with_final": 1
    }
    mnist_conf["train_rbm"]["max_iters"] = 0
    run_RBM(mnist_conf)

def test_sda_dnn():
    banner("sda dnn")
    mnist_conf = MNIST_CONF.copy()

    mnist_conf["train_sda"]["max_iters"] = MAX_ITERS
    run_SDA(mnist_conf)

    mnist_conf["train_dnn"]["max_iters"] = MAX_ITERS
    mnist_conf["init_dnn"] = {
        "filename": "temp/sda/final.nnet",
        "num_hidden_layers": -1,
        "with_final": 1
    }
    run_DNN(mnist_conf)

    mnist_conf["init_sda"] = {
        "filename": "temp/dnn/final.nnet",
        "num_hidden_layers": -1,
        "with_final": 1
    }
    mnist_conf["train_sda"]["max_iters"] = 1
    run_SDA(mnist_conf)

def test_dnn_eval():
    banner("dnn cv")
    mnist_conf = MNIST_CONF.copy()

    mnist_conf["train_dnn"]["max_iters"] = MAX_ITERS
    run_DNN(mnist_conf)

    mnist_conf["init_dnn"] = {
        "filename": "temp/dnn/final.nnet",
        "num_hidden_layers": -1,
        "with_final": 1
    }

    # per-part
    eval_DNN(mnist_conf)

    mnist_conf["eval_dnn"] = {"mode": "cv", "batch_size": 1024}
    eval_DNN(mnist_conf)

    mnist_conf["eval_dnn"] = {"mode": "per-feat", "batch_size": 1024}
    eval_DNN(mnist_conf)

def test_dropout():
    banner("dropout")
    mnist_conf = MNIST_CONF.copy()
    mnist_conf["train_dnn"]["max_iters"] = MAX_ITERS
    mnist_conf["model"]["dropout_factor"] = "0.4"
    run_DNN(mnist_conf)
