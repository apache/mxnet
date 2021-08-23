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

# Model Backwards Compatibility Tests

This folder contains the scripts that are required to run the nightly job of verifying the compatibility and inference results of models (trained on earlier versions of MXNet) when loaded on the latest release candidate. The tests flag if:
- The models fail to load on the latest version of MXNet.
- The inference results are different. 

 
## JenkinsfileForMBCC
This is configuration file for jenkins job.

## Details 
- Currently the APIs that covered for model saving/loading are : do_checkpoint/load_checkpoint, save_params/load_params, save_parameters/load_parameters(added v1.2.1 onwards), export/gluon.SymbolBlock.imports. 
- These APIs are covered over models with architectures such as : MLP, RNNs, LeNet, LSTMs covering the four scenarios described above.
- More operators/models will be added in the future to extend the operator coverage. 
- The model train file is suffixed by `_train.py` and the trained models are hosted in AWS S3.
- The trained models for now are backfilled into S3 starting from every MXNet release version v1.1.0 via the `train_mxnet_legacy_models.sh`. 
- `train_mxnet_legacy_models.sh` script checks out the previous two releases using git tag command and trains and uploads models to S3 on those MXNet versions.
- The S3 bucket's folder structure looks like this : 
    * 1.1.0/<model-1-files>  1.1.0/<model-2-files> 
    * 1.2.0/<model-1-files> 1.2.0/<model-2-files>
- The <model-1-files> is also a folder which contains the trained model symbol definitions, toy datasets it was trained on, weights and parameters of the model and other relevant files required to reload the model.
- Over a period of time, the training script would have accumulated a repository of models trained over several versions of MXNet (both major and minor releases).
- The inference part is checked via the script `model_backwards_compat_inference.sh`.
- The inference script scans the S3 bucket for MXNet version folders as described above and runs the inference code for each model folder found.

