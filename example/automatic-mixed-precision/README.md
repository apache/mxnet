<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Conversion of FP32 models to Mixed Precision Models


This folder contains examples for converting FP32 models to mixed precision models. The script allows for converting FP32 symbolic models or gluon models to mixed precision model.

## Basic Usages

1. AMP Model Conversion for a gluon model, casting the params wherever possible to FP16. The below script will convert the `resnet101_v1` model to Mixed Precision Model and cast params to FP16 wherever possible, load this converted model and run inference on it.

```bash
python amp_model_conversion.py --model resnet101_v1 --use-gluon-model  --run-dummy-inference --cast-optional-params
```

2. AMP Model Conversion for a symbolic model, keeping the params in FP32 wherever possible (--cast-optional-params not used).

```bash
python amp_model_conversion.py --model imagenet1k-resnet-152  --run-dummy-inference
```
