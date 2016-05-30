#!/usr/bin/env python
import sys
import codecs
import numpy as np

LabelVocab = {'B':0, 'M':1, 'E':2, 'S':3}

def gold_to_conll(infile):
    for line in codecs.open(infile, 'r', 'utf-8'):
        words = line.strip().split()
        for word in words:
            num_chars = len(word)
            for idx, char in enumerate(word):
                char = char.encode('utf-8')
                if num_chars == 1:
                    print '%s\t%s' % (char, 'S')
                else:
                    if idx == 0:
                        print '%s\t%s' % (char, 'B')
                    elif idx == num_chars - 1:
                        print '%s\t%s' % (char, 'E')
                    else:
                        print '%s\t%s' % (char, 'M')
        print

def load_data(infile, vocab=None, train=True):
    if vocab is None:
        vocab = {}
        vocab['#_beg_#'] = 0
        vocab['#_end_#'] = 1
        vocab['#_unknown_#'] = 2
    X_data = []
    y_data = []
    x = []
    y = []
    for line in open(infile):
        line = line.strip()
        if line == "": # begin a new sentence:
            if len(x) != 0:
                X_data.append(x)
                y_data.append(y)
                x = []
                y = []
        else:
            w, label = line.split('\t')
            y.append(LabelVocab[label])
            if w not in vocab:
                if train:
                    vocab[w] = len(vocab)
                    x.append(vocab[w])
                else:
                    x.append(vocab['#_unknown_#'])
            else:
                x.append(vocab[w])
    
    if len(x) != 0:
        X_data.append(x)
        y_data.append(y)
    return X_data, y_data, vocab

def reshape_data(sentences, labels, vocab, context_size=5, step=10):
    padding_num = int((context_size - 1) / 2)
    x = []
    y = []
    for sen, label in zip(sentences, labels):
        predict_word_num = len(sen)
        add_num = step - predict_word_num % step
        for i in range(add_num):
            sen.append(vocab['#_end_#'])
            label.append(LabelVocab['S'])

        for _ in range(padding_num):
            sen.insert(0, vocab['#_beg_#'])
            sen.append(vocab['#_end_#'])
        
        x_t = []
        y_t = []
        for i in range(padding_num, len(sen)-padding_num):
            if len(x_t) == step:
                x.append(x_t)
                y.append(y_t)
                x_t = []
                y_t = []
            x_t.append(sen[i-padding_num:i+padding_num+1])
            y_t.append(label[i-padding_num])

        if len(x_t) == step:
            x.append(x_t)
            y.append(y_t)
    
    return np.array(x), np.array(y)
    

if __name__ == '__main__':
    test_path = "test.conll"
    x, y, vocab = load_data(test_path)
    print 'vocab size %d' % (len(vocab))
    X_data, y_data = reshape_data(x, y, vocab)
    print X_data.shape, y_data.shape
    print X_data[0]
    print y_data[0]
    
