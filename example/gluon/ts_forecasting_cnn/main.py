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


"""

End to end Lorenz Map data generation and uncoditional and conditional time series prediction
model train, predict and evaluate on test using a CNN inspired from WaveNet architecture.
"""

import mxnet as mx

from model_train import Train
from model_predict import Predict
from data_generation import LorenzMapData
from data_iterator_builder import DIterators
from arg_parser import ArgParser
from eval import Evaluate

mx.random.seed(1235)

def main():
    """

    Run train and predict for the various Lorenz map prediction models with user
    provided arguments. Assets are saved in the 'assets' folder in the project directory.

    Models can be Conditional Wavenet-inspired (cw), Unconditional Wavenet-inspired (w),

    Targets to predict are x (ts=0), y(ts=1), or z(ts=2) Lorenz trajectories.
    """

    argparser = ArgParser()
    options = argparser.parse_args()
    data_generator = LorenzMapData(options)
    train_data, test_data = data_generator.generate_train_test_sets()

    # Train
    trainer = Train(options)
    train_iter = DIterators(options).build_iterator(train_data, for_train=True)
    trainer.train(train_iter)

    # Predict on test set and evaluate
    predictor = Predict(options)
    predict_iter = DIterators(options).build_iterator(test_data, for_train=False)
    predictor.predict(predict_iter)

    # Evaluate performance on test set
    evaluator = Evaluate(options)
    evaluator()

if __name__ == '__main__':
    main()
