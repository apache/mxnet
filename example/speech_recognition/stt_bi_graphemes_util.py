import csv
from collections import Counter


def split_every(n, label):
    index = 0
    if index <= len(label) - 1 <= index + n - 1:
        yield label[index:len(label)]
        index = index + n
    while index+n-1 <= len(label)-1:
        yield label[index:index+n]
        index = index + n
        if index <= len(label)-1 <= index+n-1:
            yield label[index:len(label)]
            index=index+n

def generate_bi_graphemes_label(label):
    label_bi_graphemes = []
    label = label.split(' ')
    last_index = len(label) - 1
    for label_index, item in enumerate(label):
        for pair in split_every(2, item):
            label_bi_graphemes.append(pair)
        if label_index != last_index:
            label_bi_graphemes.append(" ")
    return label_bi_graphemes

def generate_bi_graphemes_dictionary(label_list):
    freqs = Counter()
    for label in label_list:
        label = label.split(' ')
        for i in label:
            for pair in split_every(2, i):
                if len(pair) == 2:
                    freqs[pair] += 1


    with open('resources/unicodemap_en_baidu_bi_graphemes.csv', 'w') as bigram_label:
        bigramwriter = csv.writer(bigram_label, delimiter = ',')
        baidu_labels = list('\' abcdefghijklmnopqrstuvwxyz')
        for index, key in enumerate(baidu_labels):
            bigramwriter.writerow((key, index+1))
        for index, key in enumerate(freqs.keys()):
            bigramwriter.writerow((key, index+len(baidu_labels)+1))
