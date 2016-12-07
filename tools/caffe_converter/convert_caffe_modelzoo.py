import os
import requests
import argparse
import logging
from convert_model import convert_model
from convert_mean import convert_mean

_default_model_info = {
    'bvlc_alexnet' : {
        'prototxt' : 'https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt',
        'caffemodel' : 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel',
        'mean' : 'https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/caffe/imagenet_mean.binaryproto',
        'top-1-acc' : 0.571,
        'top-5-acc' : 0.802
    },
    'bvlc_googlenet' : {
        'prototxt' : 'https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt',
        'caffemodel' : 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel',
        'mean' : (123,117,104),
        'top-1-acc' : 0.687,
        'top-5-acc' : 0.889
    },
    'vgg-16' : {
        'prototxt' : 'https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/c3ba00e272d9f48594acef1f67e5fd12aff7a806/VGG_ILSVRC_16_layers_deploy.prototxt',
        'caffemodel' : 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel' ,
        'mean': (123.68,116.779,103.939),
        'top-1-acc' : 0.734,
        'top-5-acc' : 0.914
    },
    'vgg-19' : {
        'prototxt' : 'https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt',
        'caffemodel' : 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel',
        'mean' : (123.68,116.779,103.939),
        'top-1-acc' : 0.731,
        'top-5-acc' : 0.913
    },
    'resnet-50' : {
        'prototxt' : 'https://raw.githubusercontent.com/KaimingHe/deep-residual-networks/master/prototxt/ResNet-50-deploy.prototxt',
        'caffemodel' : 'https://iuxblw-bn1306.files.1drv.com/y3mK98PDQUb4kEKq_HxeBlaazrkKkwHNKmsdXpj-mjwBevTrH_x3q-X1m1VPCDREIn_Iwj8yovo38aUEH3YY2Q_8mQpSjA4i1_Cbc3HdRf2JpS5XtaITvTAf4HzbSh2oD2hnNsjwEUXVbDUgS_PmUtsky7-mb_dr5YaXh2UUZmr4Ew/ResNet-50-model.caffemodel',
        'mean' : 'https://iuxblw-bn1306.files.1drv.com/y3mBJiunDBrRxGRoPOX_SEq9o2qolrlrDpgpGGQO80Wq3q3lD-8xB6G6RdLnMRFmoJ_zQ6q7GBdp46wj_nCw0chy88Tkao3A9nasYyRpjtME2Dwl6qe-Rz9W2AfIYoID3e479PXUAEQiB_2yQKF2T5masHtU9yXYX1-2Z8rY-PhNyw/ResNet_mean.binaryproto',
        'top-1-acc' : 0.753,
        'top-5-acc' : 0.922
    },
    'resnt-101' : {
        'prototxt' : 'https://raw.githubusercontent.com/KaimingHe/deep-residual-networks/master/prototxt/ResNet-101-deploy.prototxt',
        'caffemodel' : 'https://iuxblw-bn1306.files.1drv.com/y3mST4ljYyU_XRPxIrzzrQDyRZj5sSr25jWuZZad9QUnw778z8IN-W4yy9xW1lWleE1Ejrlz5D7jvChlMts5E3MKy9K0rtajRj8V3o_Mne_MGRR2ExVEeRubngO5zCYSMHZs9Nupwasf8ZjpWDcDxcAADFtBAG4tyUakY6mBN2wjdo/ResNet-101-model.caffemodel',
        'mean' : 'https://iuxblw-bn1306.files.1drv.com/y3mBJiunDBrRxGRoPOX_SEq9o2qolrlrDpgpGGQO80Wq3q3lD-8xB6G6RdLnMRFmoJ_zQ6q7GBdp46wj_nCw0chy88Tkao3A9nasYyRpjtME2Dwl6qe-Rz9W2AfIYoID3e479PXUAEQiB_2yQKF2T5masHtU9yXYX1-2Z8rY-PhNyw/ResNet_mean.binaryproto',
        'top-1-acc' : 0.764,
        'top-5-acc' : 0.929
    },
    'resnet-152' : {
        'prototxt' : 'https://raw.githubusercontent.com/KaimingHe/deep-residual-networks/master/prototxt/ResNet-152-deploy.prototxt',
        'caffemodel' : 'https://iuxblw-bn1306.files.1drv.com/y3m13PGQQ3prKb-1GhTOqwRuS88CLy0O0F5bZfsNMklAzwpvM4SchXXPBhb3lDLgF0Tc-q4HHcTgsfv7ipRCTB_Y8qBl86NJzsLNF8kmu3YBSk-XfltvPA4hKqKttCKauFN6QCBDiHV3LPH790q9sNRF-BnE5O4dk9zYA1q19u80LI/ResNet-152-model.caffemodel',
        'mean' : 'https://iuxblw-bn1306.files.1drv.com/y3mBJiunDBrRxGRoPOX_SEq9o2qolrlrDpgpGGQO80Wq3q3lD-8xB6G6RdLnMRFmoJ_zQ6q7GBdp46wj_nCw0chy88Tkao3A9nasYyRpjtME2Dwl6qe-Rz9W2AfIYoID3e479PXUAEQiB_2yQKF2T5masHtU9yXYX1-2Z8rY-PhNyw/ResNet_mean.binaryproto',
        'top-1-acc' : 0.77,
        'top-5-acc' : 0.933
    },
}

def get_model_info(model_name):
    return dict(dict(_default_model_info)[model_name])

def download_file(url, local_fname=None, force_write=False):
    if local_fname is None:
        local_fname = url.split('/')[-1]
    if not force_write and os.path.exists(local_fname):
        return local_fname

    r = requests.get(url, stream=True)
    assert r.status_code == 200, "failed to open %s" % url
    with open(local_fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    return local_fname

def download_caffe_model(model_name, dst_dir, meta_info):
    meta_info = dict(meta_info)
    assert model_name in meta_info
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    meta = dict(meta_info[model_name])
    model_name = os.path.join(dst_dir, model_name)
    assert 'prototxt' in meta, "missing prototxt url"
    prototxt = download_file(meta['prototxt'], model_name+'_deploy.prototxt')
    assert 'caffemodel' in meta, "mssing caffemodel url"
    caffemodel = download_file(meta['caffemodel'], model_name+'.caffemodel')
    assert 'mean' in meta, 'no mean info'
    mean = meta['mean']
    if isinstance(mean, str):
        mean = download_file(mean, model_name+'_mean.binaryproto')
    return (prototxt, caffemodel, mean)

def convert_caffe_model(model_name, dst_dir='./model', meta_info=None):
    if meta_info is None:
        meta_info = _default_model_info
    (prototxt, caffemodel, mean) = download_caffe_model(model_name, dst_dir, meta_info)
    model_name = os.path.join(dst_dir, model_name)
    convert_model(prototxt, caffemodel, model_name)
    if isinstance(mean, str):
        out = model_name + '-mean.nd'
        convert_mean(mean, out)
        mean = out
    return (model_name, mean)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Caffe model zoo')
    parser.add_argument('model_name', help='can be '+', '.join(_default_model_info.keys()))
    args = parser.parse_args()
    convert_caffe_model(args.model_name)
