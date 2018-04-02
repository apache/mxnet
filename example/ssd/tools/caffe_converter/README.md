# Convert Caffe Model to Mxnet Format

This folder contains the source codes for this tool.

If Caffe with python binding is installed, we can use the following command to
convert a Resnet-50 pretrained model.

```bash
python convert_caffe_modelzoo.py resnet-50
```

Please refer to
[docs/faq/caffe.md](../../docs/faq/caffe.md) for more details.

### How to use
To convert ssd caffemodels, Use: `python convert_model.py prototxt caffemodel outputprefix`

### Note

Use this converter for ssd caffemodels only. General converter is available in `mxnet/tools/caffe_converter`.
