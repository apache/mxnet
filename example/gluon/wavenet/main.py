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
    Description : run code using argument info
    """
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoches', type=int, default=10)
    parser.add_argument('--mu', type=int, default=128)
    parser.add_argument('--n_residue', type=int, default=24)
    parser.add_argument('--n_skip', type=int, default=128)
    parser.add_argument('--dilation_depth', type=int, default=10)
    parser.add_argument('--n_repeat', type=int, default=2)
    parser.add_argument('--seq_size', type=int, default=20000)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--generation', type=bool, default=True)
    config = parser.parse_args()

    trainer = Train(config)

    trainer.train()
    if config.generation:
        trainer.generation()

if __name__ == "__main__":
    main()
