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

from pathlib import Path
from functools import partial

import details
from neural_compressor.experimental import Quantization, common

# constants
INC_CONFIG_PATH = Path('./bert.yaml').resolve()
PARAMS_PATH = Path('./bert_mrpc.params').resolve()
OUTPUT_DIR_PATH = Path('./output/').resolve()
OUTPUT_MODEL_PATH = OUTPUT_DIR_PATH/'quantized_model'
OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)

# Prepare the dataloaders (calib_dataloader is same as train_dataloader but without shuffling)
train_dataloader, dev_dataloader, calib_dataloader = details.preprocess_data()

# Get the model
model = details.BERTModel(details.BACKBONE, dropout=0.1, num_classes=details.NUM_CLASSES)
model.hybridize(static_alloc=True)

# finetune or load the parameters of already finetuned model
if not PARAMS_PATH.exists():
    model = details.finetune(model, train_dataloader, dev_dataloader, OUTPUT_DIR_PATH)
    model.save_parameters(str(PARAMS_PATH))
else:
    model.load_parameters(str(PARAMS_PATH), ctx=details.CTX, cast_dtype=True)

# run INC
calib_dataloader.batch_size = details.BATCH_SIZE
eval_func = partial(details.evaluate, dataloader=dev_dataloader)

quantizer = Quantization(str(INC_CONFIG_PATH))  # 1. Config file
quantizer.model = common.Model(model)           # 2. Model to be quantized
quantizer.calib_dataloader = calib_dataloader   # 3. Calibration dataloader
quantizer.eval_func = eval_func                 # 4. Evaluation function
quantized_model = quantizer.fit().model

# save the quantized model
quantized_model.export(str(OUTPUT_MODEL_PATH))
