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


import shutil, random, glob, os, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

splits = ('train', 'validation', 'test')
targets = ('healthy', 'diseased')

def split_file_list(file_list, train_split=0.7, val_split=0.2, test_split=0.2):
    random.shuffle(file_list)
    files = len(file_list)
    train_items = round(files * train_split)
    validation_items = round(files * val_split)
    train = file_list[:train_items]
    validation = file_list[train_items: train_items + validation_items]
    test = file_list[train_items + validation_items:]

    return train, validation, test

def process_dataset(root_directory, splits=splits, classes=targets, train=0.7, val=0.2, test=0.2):

    # Get healthy and diseased file lists
    for target in targets:
        file_list = glob.glob(f"{root_directory}/**/{target}/*.JPG")
        dataset_splits = split_file_list(file_list, train, val, test)
        logger.info(f"Starting transferring files from the {target} class")
        for idx, split in enumerate(dataset_splits):
            new_path = os.path.join("datasets", splits[idx], target)
            logger.info(f"Moving {splits[idx]} files")
            Path(new_path).mkdir(parents=True, exist_ok=True)
            for file_path in split:
                shutil.move(file_path, new_path)
            logger.info(f"Finished moving {splits[idx]} files")
    logger.info(f"Finished moving files")
    logger.info("Removing old folders")
    shutil.rmtree(root_directory)
    logger.info("Finished!")
