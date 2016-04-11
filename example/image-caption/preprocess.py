# -*- coding:utf-8 -*-
# @author: Yuanqin Lu

import os
import h5py
import numpy as np
import json
from scipy.misc import imread, imresize


"""
coco_raw.json
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ',
u'Man riding a motor bike on a dirt road on the countryside.',
u'A man riding on the back of a motorcycle.',
u'A dirt path with a young person on a motor bike rests to the foreground
of a verdant area with a bridge and a background of cloud-wreathed mountains. ',
u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'],
'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

"""

def transform_img(input_json, output_h5, output_json, image_root):
    imgs = json.load(open(input_json, 'r'))
    N = len(imgs)
    f = h5py.File(output_h5, 'w')
    dset = f.create_dataset("images", (N, 3, 256, 256), dtype='uint8')
    for i, img in enumerate(imgs):
        I = imread(os.path.join(image_root, img['file_path']))
        try:
            Ir = imresize(I, (256, 256))
        except:
            print 'failed resizeing image %s' % img['file_path']
            raise
        if len(Ir.shape) == 2:
            Ir = Ir[:, :, np.newaxis]
            Ir = np.concatenate((Ir, Ir, Ir), axis=2)
        Ir = Ir.transpose(2, 0, 1)
        dset[i] = Ir
        img['img_idx'] = i
        if i % 1000 == 0:
            print 'processing %d / %d (%.2f%% done)' % (i, N, i * 100.0 / N)
    f.close()
    print 'wrote ', output_h5
    json.dump(imgs, open(output_json, 'w'))
    print 'wrote ', output_json



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, help='path of json file with file path and captions')
    parser.add_argument('--output_json', type=str, help='path to save output json file')
    parser.add_argument('--output_hdf5', type=str, help='path to save output hdf5 file')
    parser.add_argument('--image_root', type=str, help='path of image root dir')

    args = parser.parse_args()
    transform_img(args.input_json, args.output_hdf5, args.output_json, args.image_root)

