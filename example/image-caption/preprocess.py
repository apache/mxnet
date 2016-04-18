"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image,
  such as in particular the 'split' it was assigned to.
"""

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize


def prepro_captions(imgs):
    # preprocess all the captions
    print 'example processed tokens:'
    for i, img in enumerate(imgs):
        img['processed_tokens'] = []
        for j, s in enumerate(img['captions']):
            txt = str(s).lower().translate(None, string.punctuation).strip().split()
            img['processed_tokens'].append(txt)
            if i < 10 and j == 0: print txt


def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for txt in img['processed_tokens']:
            for w in txt:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.iteritems()], reverse=True)
    print 'top words and their counts:'
    print '\n'.join(map(str, cw[:20]))

    # print some stats
    total_words = sum(counts.itervalues())
    print 'total words:', total_words
    bad_words = [w for w, n in counts.iteritems() if n <= count_thr]
    vocab = [w for w, n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts))
    print 'number of words in vocab would be %d' % (len(vocab),)
    print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words)

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for txt in img['processed_tokens']:
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print 'max length sentence in raw data: ', max_len
    print 'sentence length distribution (count, number of words):'
    sum_len = sum(sent_lengths.values())
    for i in xrange(max_len + 1):
        print '%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len)

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print 'inserting the special UNK token'
        vocab.append('UNK')

    for img in imgs:
        img['final_captions'] = []
        for txt in img['processed_tokens']:
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            img['final_captions'].append(caption)

    return vocab


def encode_captions(imgs, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs)  # total number of captions

    label_arrays = []
    idx_pairs = np.zeros(M, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    for i, img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')
        for j, s in enumerate(img['final_captions']):
            label_length[caption_counter] = min(max_length, len(s))  # record the length of this sequence
            idx_pairs[caption_counter] = i
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]


        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)

    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print 'encoded captions to array of size ', `L.shape`
    return L, idx_pairs, label_length


def main(params):
    imgs = json.load(open(params['input_json'], 'r'))
    seed(123)  # make reproducible
    shuffle(imgs)  # shuffle the order

    # tokenization and preprocessing
    prepro_captions(imgs)

    # create the vocab
    vocab = build_vocab(imgs, params)
    itow = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

    # assign the splits
    if not imgs[0].has_key('split'):
        assign_splits(imgs, params)

    # encode captions in large arrays, ready to ship to hdf5 file
    L, idx_pairs, label_length = encode_captions(imgs, params, wtoi)

    # create output h5 file
    N = len(imgs)
    f = h5py.File(params['output_h5'], "w")
    f.create_dataset("labels", dtype='uint32', data=L)
    f.create_dataset("idx_pairs", dtype='uint32', data=idx_pairs)
    f.create_dataset("label_length", dtype='uint32', data=label_length)
    dset = f.create_dataset("images", (N, 3, 224, 224), dtype='uint8')  # space for resized images
    for i, img in enumerate(imgs):
        # load the image
        I = imread(os.path.join(params['images_root'], img['file_path']))
        try:
            Ir = imresize(I, (224, 224))
        except:
            print 'failed resizing image %s - see http://git.io/vBIE0' % (img['file_path'],)
            raise
        # handle grayscale input images
        if len(Ir.shape) == 2:
            Ir = Ir[:, :, np.newaxis]
            Ir = np.concatenate((Ir, Ir, Ir), axis=2)
        # and swap order of axes from (256,256,3) to (3,256,256)
        Ir = Ir.transpose(2, 0, 1)
        # write to h5
        dset[i] = Ir
        if i % 1000 == 0:
            print 'processing %d/%d (%.2f%% done)' % (i, N, i * 100.0 / N)
    f.close()
    print 'wrote ', params['output_h5']

    # create output json file
    out = {}
    out['ix_to_word'] = itow  # encode the (1-indexed) vocab
    out['images'] = []
    for i, img in enumerate(imgs):
        jimg = {}
        if 'file_path' in img: jimg['file_path'] = img['file_path']  # copy it over, might need
        if 'id' in img: jimg['id'] = img['id']  # copy over & mantain an id, if present (e.g. coco ids, useful)

        out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print 'wrote ', params['output_json']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--output_json', default='data.json', help='output json file')
    parser.add_argument('--output_h5', default='data.h5', help='output h5 file')

    # options
    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--images_root', default='',
                        help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent=2)
    main(params)
