import os, zipfile
import mxnet
from mxnet.test_utils import download

def unzip_file(filename, outpath):
    fh = open(filename, 'rb')
    z = zipfile.ZipFile(fh)
    for name in z.namelist():
        z.extract(name, outpath)
    fh.close()

download('http://msvocds.blob.core.windows.net/coco2014/train2014.zip', 'dataset/train2014.zip')
download('http://msvocds.blob.core.windows.net/coco2014/val2014.zip', 'dataset/val2014.zip')

unzip_file('dataset/train2014.zip', 'dataset')
unzip_file('dataset/val2014.zip', 'dataset')
