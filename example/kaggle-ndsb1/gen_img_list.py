import csv
import os
import sys
import random

if len(sys.argv) < 4:
    print "Usage: gen_img_list.py train/test sample_submission.csv train_folder img.lst"
    exit(1)

random.seed(888)

task = sys.argv[1]
fc = csv.reader(file(sys.argv[2]))
fi = sys.argv[3]
fo = csv.writer(open(sys.argv[4], "w"), delimiter='\t', lineterminator='\n')

# make class map
head = fc.next()
head = head[1:]

# make image list
img_lst = []
cnt = 0
if task == "train":
    for i in xrange(len(head)):
        path = fi + head[i]
        lst = os.listdir(fi + head[i])
        for img in lst:
            img_lst.append((cnt, i, path + '/' + img))
            cnt += 1
else:
    lst = os.listdir(fi)
    for img in lst:
        img_lst.append((cnt, 0, fi + img))
        cnt += 1

# shuffle
random.shuffle(img_lst)

#wirte
for item in img_lst:
    fo.writerow(item)

