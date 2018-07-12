# Model Backwards Compatibility Tests

This folder contains the scripts that are required to run the nightly job of verifying the compatibility and inference results of models (trained on earlier versions of MXNet) when loaded on the latest release candidate. The tests flag if:
- The models fail to load on the latest version of MXNet.
- The inference results are different. 

 
## JenkinsfileForMBCC
This is configuration file for jenkins job.

## Details
- The `model_backward_compat_checker.sh` is a top level script that invokes the inference files in python. 
- Currently the APIs that covered for model saving/loading are : do_checkpoint/load_checkpoint, save_params/load_params, save_parameters/load_parameters(added v1.2.1 onwards), export/gluon.SymbolBlock.imports. 
- These APIs are covered over models with architectures such as : MLP, RNNs, LeNet covering the four scenarios described above.
- More operators/models will be added in the future to extend the operator coverage. 
- The model train file is suffixed by `_train.py` and the trained models are hosted in AWS S3.
- The trained models for now are backfilled into S3 starting from every MXNet release version v1.1.0.
- The script for training the models on older versions of MXNet is : `train_mxnet_legacy_models.sh`.
- The inference file is suffixed by `_inference.py`.

