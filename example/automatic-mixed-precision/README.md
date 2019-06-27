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
