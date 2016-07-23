import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../python"))
import mxnet as mx
import random
import argparse
import cv2
import time


def list_image(root, recursive, exts):
    image_list = []
    if recursive:
        cat = {}
        for path, subdirs, files in os.walk(root, followlinks=True):
            subdirs.sort()
            print(len(cat), path)
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    if path not in cat:
                        cat[path] = len(cat)
                    image_list.append((len(image_list), os.path.relpath(fpath, root), cat[path]))
    else:
        for fname in os.listdir(root):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                image_list.append((len(image_list), os.path.relpath(fpath, root), 0))
    return image_list


def write_list(path_out, image_list):
    with open(path_out, 'w') as fout:
        n_images = xrange(len(image_list))
        for i in n_images:
            line = '%d\t' % image_list[i][0]
            for j in image_list[i][2:]:
                line += '%d\t' % j
            line += '%s\n' % image_list[i][1]
            fout.write(line)


def make_list(args):
    image_list = list_image(args.root, args.recursive, args.exts)
    if args.shuffle is True:
        random.seed(100)
        random.shuffle(image_list)
    N = len(image_list)
    chunk_size = (N + args.chunks - 1) / args.chunks
    for i in xrange(args.chunks):
        chunk = image_list[i * chunk_size:(i + 1) * chunk_size]
        if args.chunks > 1:
            str_chunk = '_%d' % i
        else:
            str_chunk = ''
        sep = int(chunk_size * args.train_ratio)
        sep_test = int(chunk_size * args.test_ratio)
        write_list(args.prefix + str_chunk + '_test.lst', chunk[:sep_test])
        write_list(args.prefix + str_chunk + '_train.lst', chunk[sep_test:sep_test + sep])
        write_list(args.prefix + str_chunk + '_val.lst', chunk[sep_test + sep:])


def read_list(path_in):
    image_list = []
    with open(path_in) as fin:
        for line in fin.readlines():
            line = [i.strip() for i in line.strip().split('\t')]
            item = [int(line[0])] + [line[-1]] + [int(i) for i in line[1:-1]]
            image_list.append(item)
    return image_list


def write_record(args, image_list, fname):
    source = image_list
    tic = [time.time()]
    color_modes = {-1: cv2.IMREAD_UNCHANGED,
                   0: cv2.IMREAD_GRAYSCALE,
                   1: cv2.IMREAD_COLOR}
    total = len(source)

    def image_encode(item, q_out):
        try:
            img = cv2.imread(os.path.join(args.root, item[1]), color_modes[args.color])
        except:
            print 'imread error:', item[1]
            return
        if img is None:
            print 'read none error:', item[1]
            return
        if args.center_crop:
            if img.shape[0] > img.shape[1]:
                margin = (img.shape[0] - img.shape[1]) / 2;
                img = img[margin:margin + img.shape[1], :]
            else:
                margin = (img.shape[1] - img.shape[0]) / 2;
                img = img[:, margin:margin + img.shape[0]]
        if args.resize:
            if img.shape[0] > img.shape[1]:
                newsize = (args.resize, img.shape[0] * args.resize / img.shape[1])
            else:
                newsize = (img.shape[1] * args.resize / img.shape[0], args.resize)
            img = cv2.resize(img, newsize)
        header = mx.recordio.IRHeader(0, item[2], item[0], 0)

        try:
            s = mx.recordio.pack_img(header, img, quality=args.quality, img_fmt=args.encoding)
            q_out.put(('data', s, item))
        except:
            print 'pack_img error:', item[1]
            return

    def read_worker(q_in, q_out):
        while not q_in.empty():
            item = q_in.get()
            image_encode(item, q_out)

    def write_worker(q_out, fname, saving_folder):
        pre_time = time.time()
        sink = []
        os.chdir(saving_folder)
        fname_rec = fname[:fname.rfind('.')]
        record = mx.recordio.MXRecordIO(fname_rec + '.rec', 'w')
        while True:
            stat, s, item = q_out.get()
            if stat == 'finish':
                write_list(fname_rec + '.lst', sink)
                break
            record.write(s)
            sink.append(item)
            if len(sink) % 1000 == 0:
                cur_time = time.time()
                print 'time:', cur_time - pre_time, ' count:', len(sink)
                pre_time = cur_time

    try:
        import multiprocessing
        q_in = [multiprocessing.Queue() for i in range(args.num_thread)]
        q_out = multiprocessing.Queue(1024)
        for i in range(len(image_list)):
            q_in[i % len(q_in)].put(image_list[i])
        read_process = [multiprocessing.Process(target=read_worker, args=(q_in[i], q_out)) \
                        for i in range(args.num_thread)]
        for p in read_process:
            p.start()
        write_process = multiprocessing.Process(target=write_worker, args=(q_out, fname, args.saving_folder))
        write_process.start()
        for p in read_process:
            p.join()
        q_out.put(('finish', '', []))
        write_process.join()
    except ImportError:
        print('multiprocessing not available, fall back to single threaded encoding')
        import Queue
        q_out = Queue.Queue()
        os.chdir(args.saving_folder)
        fname_rec = fname[:fname.rfind('.')]
        record = mx.recordio.MXRecordIO(fname_rec + '.rec', 'w')
        cnt = 0
        pre_time = time.time()
        for item in image_list:
            image_encode(item, q_out)
            if q_out.empty():
                continue
            _, s, _ = q_out.get()
            record.write(s)
            cnt += 1
            if cnt % 1000 == 0:
                cur_time = time.time()
                print 'time:', cur_time - pre_time, ' count:', cnt
                pre_time = cur_time


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('prefix', help='prefix of input/output files.')
    parser.add_argument('root', help='path to folder containing images.')

    cgroup = parser.add_argument_group('Options for creating image lists')
    cgroup.add_argument('--list', type=bool, default=False,
                        help='If this is set im2rec will create image list(s) by traversing root folder\
        and output to <prefix>.lst.\
        Otherwise im2rec will read <prefix>.lst and create a database at <prefix>.rec')
    cgroup.add_argument('--exts', type=list, default=['.jpeg', '.jpg'],
                        help='list of acceptable image extensions.')
    cgroup.add_argument('--chunks', type=int, default=1, help='number of chunks.')
    cgroup.add_argument('--train-ratio', type=float, default=1.0,
                        help='Ratio of images to use for training.')
    cgroup.add_argument('--test-ratio', type=float, default=0,
                        help='Ratio of images to use for testing.')
    cgroup.add_argument('--recursive', type=bool, default=False,
                        help='If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.')

    rgroup = parser.add_argument_group('Options for creating database')
    rgroup.add_argument('--resize', type=int, default=0,
                        help='resize the shorter edge of image to the newsize, original images will\
        be packed by default.')
    rgroup.add_argument('--center-crop', type=bool, default=False,
                        help='specify whether to crop the center image to make it rectangular.')
    rgroup.add_argument('--quality', type=int, default=80,
                        help='JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9')
    rgroup.add_argument('--num_thread', type=int, default=1,
                        help='number of thread to use for encoding. order of images will be different\
        from the input list if >1. the input list will be modified to match the\
        resulting order.')
    rgroup.add_argument('--color', type=int, default=1, choices=[-1, 0, 1],
                        help='specify the color mode of the loaded image.\
        1: Loads a color image. Any transparency of image will be neglected. It is the default flag.\
        0: Loads image in grayscale mode.\
        -1:Loads image as such including alpha channel.')
    rgroup.add_argument('--encoding', type=str, default='.jpg', choices=['.jpg', '.png'],
                        help='specify the encoding of the images.')
    rgroup.add_argument('--saving-folder', type=str, default='.',
                        help='folder in which .rec files will be saved.')
    rgroup.add_argument('--shuffle', default=True, help='If this is set as True, \
        im2rec will randomize the image order in <prefix>.lst')
    args = parser.parse_args()
    if args.list:
        make_list(args)
    else:
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        for f in files:
            if f.startswith(args.prefix) is True and f.endswith('.lst') is True:
                print 'Creating .rec file from', f, 'in', args.saving_folder
                image_list = read_list(f)
                write_record(args, image_list, f)


if __name__ == '__main__':
    main()
