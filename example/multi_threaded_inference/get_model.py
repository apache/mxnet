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
import argparse
import mxnet as mx
import gluoncv


models = ["imagenet1k-inception-bn", "imagenet1k-resnet-50",
          "imagenet1k-resnet-152", "imagenet1k-resnet-18"]

def main():
    logging.basicConfig()
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description='Download model hybridize and save as symbolic model for multithreaded inference')
    parser.add_argument("--model", type=str, choices=models, required=True)
    args = parser.parse_args()

    mx.test_utils.download_model(args.model)

if __name__ == "__main__":
    main()
