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
Description : main module to run the lipnet training code
"""


import argparse
from trainer import Train

def main():
    """
    Description : run lipnet training code using argument info
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--image_path', type=str, default='./data/datasets/')
    parser.add_argument('--align_path', type=str, default='./data/align/')
    parser.add_argument('--dr_rate', type=float, default=0.5)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--model_path', type=str, default=None)
    config = parser.parse_args()
    trainer = Train(config)
    trainer.build_model(dr_rate=config.dr_rate, path=config.model_path)
    trainer.load_dataloader()
    trainer.run(epochs=config.epochs)

if __name__ == "__main__":
    main()
    