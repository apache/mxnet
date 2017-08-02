# Convert MXNet models into Apple CoreML format.

This tool helps convert MXNet models into Apple CoreML format which can then be run on Apple devices.

In order to use this tool you need to have these installed:
mxnet 0.10.0
python 2.7
coremltools 0.4.0
```bash
pip install -U coremltools
```
_Note: -U option is currently giving issues. Use 'pip install coremltools' instead._

You can use the following command to convert an existing squeezenet-v1.1 pretrained model.

```bash
TODO
```

For some models there may not be a one-to-one correspondence with CoreML and the converter will fail if you are converting such models.
If you understand the risks with the model conversion, you can provide a "force" flag to ask converter to force convert.

#### This tool has been tested on:
High Sierra 10.13 MacOS
Xcode 9 beta 2