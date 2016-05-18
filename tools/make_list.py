import os
import random
import numpy as np
import argparse

def list_image(root, recursive, exts):
    image_list = []
    if recursive:
        cat = {}
        for path, subdirs, files in os.walk(root):
            print path
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    if path not in cat:
                        cat[path] = len(cat)
                    image_list.append((os.path.relpath(fpath, root), cat[path]))
    else:
        for fname in os.listdir(root):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                image_list.append((os.path.relpath(fpath, root), 0))
    return image_list

def write_list(path_out, image_list):
    with open(path_out, 'w') as fout:
        for i in xrange(len(image_list)):
            fout.write('%d\t%d\t%s\n'%(i, image_list[i][1], image_list[i][0]))


def make_list(prefix_out, root, recursive, exts, num_chunks, train_ratio):
    image_list = list_image(root, recursive, exts)
    random.shuffle(image_list)
    N = len(image_list)
    chunk_size = (N+num_chunks-1)/num_chunks
    for i in xrange(num_chunks):
        chunk = image_list[i*chunk_size:(i+1)*chunk_size]
        if num_chunks > 1:
            str_chunk = '_%d'%i
        else:
            str_chunk = ''
        if train_ratio < 1:
            sep = int(chunk_size*train_ratio)
            write_list(prefix_out+str_chunk+'_train.lst', chunk[:sep])
            write_list(prefix_out+str_chunk+'_val.lst', chunk[sep:])
        else:
            write_list(prefix_out+str_chunk+'.lst', chunk)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Make image list files that are\
        required by im2rec')
    parser.add_argument('root', help='path to folder that contain images.')
    parser.add_argument('prefix', help='prefix of output list files.')
    parser.add_argument('--exts', type=str, nargs='+', default=['.jpeg','.jpg'],
        help='list of acceptable image extensions.')
    parser.add_argument('--chunks', type=int, default=1, help='number of chunks.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
        help='Percent of images to use for training.')
    parser.add_argument('--recursive', type=bool, default=False,
        help='If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.')
    args = parser.parse_args()
    
    make_list(args.prefix, args.root, args.recursive,
        args.exts, args.chunks, args.train_ratio)

if __name__ == '__main__':
    main()
