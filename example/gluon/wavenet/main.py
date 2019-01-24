"""
Descrition : main module to run code
"""
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
import argparse
from trainer import Train

def main():
    """
    Description : run wavenet code using argument info
    - batch_size : Define batch size (default=64)
    - epochs : Define the total number of epochs (default=1000)
    - mu : Define mu value for mu-law algorithm (default=128)
    - n_residue : Define number of residue (default=24)
    - n_skip : Define number of skip (default=128)
    - dilation_depth : Define dilation depth (default=10)
    - use_gpu : whether or not to use the GPU (default=True)
    - generation : whether or not to generate a wave file for model (default=True)
    - load_file : file name in loading wave file
    - save_file : file name in saving result
    """
    parser = argparse.ArgumentParser(description='argument for wavenet hyperparameters')
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--epochs', type=int, default=10, help="total number of epochs")
    parser.add_argument('--mu', type=int, default=128, help="mu value for mu-law algorithm")
    parser.add_argument('--n_residue', type=int, default=24, help="number of residue")
    parser.add_argument('--n_skip', type=int, default=128, help="number of skip")
    parser.add_argument('--dilation_depth', type=int, default=10, help="number of dilation depth")
    parser.add_argument('--n_repeat', type=int, default=2, help="number of repeat")
    parser.add_argument('--seq_size', type=int, default=20000, help="number of sequence size")
    parser.add_argument('--use_gpu', type=str, default="True", help="use gpu")
    parser.add_argument('--generation', type=bool, default=True, help="generate a wave file")
    parser.add_argument('--load_file', type=str, default='parametric-2.wav', help="file name in loading wave file")
    parser.add_argument('--save_file', type=str, default='wav.npy', help="file name in saving result")
    config = parser.parse_args()
    print(config)

    trainer = Train(config)

    trainer.train()
    if config.generation:
        trainer.generation()

if __name__ == "__main__":
    main()
