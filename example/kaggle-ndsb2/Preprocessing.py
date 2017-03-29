"""Preprocessing script.

This script walks over the directories and dump the frames into a csv file
"""
import os
import csv
import sys
import random
import scipy
import numpy as np
import dicom
from skimage import io, transform
from joblib import Parallel, delayed
import dill

def mkdir(fname):
   try:
       os.mkdir(fname)
   except:
       pass

def get_frames(root_path):
   """Get path to all the frame in view SAX and contain complete frames"""
   ret = []
   for root, _, files in os.walk(root_path):
       root=root.replace('\\','/')
       files=[s for s in files if ".dcm" in s]
       if len(files) == 0 or not files[0].endswith(".dcm") or root.find("sax") == -1:
           continue
       prefix = files[0].rsplit('-', 1)[0]
       fileset = set(files)
       expected = ["%s-%04d.dcm" % (prefix, i + 1) for i in range(30)]
       if all(x in fileset for x in expected):
           ret.append([root + "/" + x for x in expected])
   # sort for reproduciblity
   return sorted(ret, key = lambda x: x[0])


def get_label_map(fname):
   labelmap = {}
   fi = open(fname)
   fi.readline()
   for line in fi:
       arr = line.split(',')
       labelmap[int(arr[0])] = line
   return labelmap


def write_label_csv(fname, frames, label_map):
   fo = open(fname, "w")
   for lst in frames:
       index = int(lst[0].split("/")[3])
       if label_map is not None:
           fo.write(label_map[index])
       else:
           fo.write("%d,0,0\n" % index)
   fo.close()


def get_data(lst,preproc):
   data = []
   result = []
   for path in lst:
       f = dicom.read_file(path)
       img = preproc(f.pixel_array.astype(float) / np.max(f.pixel_array))
       dst_path = path.rsplit(".", 1)[0] + ".64x64.jpg"
       scipy.misc.imsave(dst_path, img)
       result.append(dst_path)
       data.append(img)
   data = np.array(data, dtype=np.uint8)
   data = data.reshape(data.size)
   data = np.array(data, dtype=np.str_)
   data = data.reshape(data.size)
   return [data,result]


def write_data_csv(fname, frames, preproc):
   """Write data to csv file"""
   fdata = open(fname, "w")
   dr = Parallel()(delayed(get_data)(lst,preproc) for lst in frames)
   data,result = zip(*dr)
   for entry in data:
      fdata.write(','.join(entry)+'\r\n')
   print("All finished, %d slices in total" % len(data))
   fdata.close()
   result = np.ravel(result)
   return result


def crop_resize(img, size):
   """crop center and resize"""
   if img.shape[0] < img.shape[1]:
       img = img.T
   # we crop image from center
   short_egde = min(img.shape[:2])
   yy = int((img.shape[0] - short_egde) / 2)
   xx = int((img.shape[1] - short_egde) / 2)
   crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
   # resize to 64, 64
   resized_img = transform.resize(crop_img, (size, size))
   resized_img *= 255
   return resized_img.astype("uint8")


def local_split(train_index):
   random.seed(0)
   train_index = set(train_index)
   all_index = sorted(train_index)
   num_test = int(len(all_index) / 3)
   random.shuffle(all_index)
   train_set = set(all_index[num_test:])
   test_set = set(all_index[:num_test])
   return train_set, test_set


def split_csv(src_csv, split_to_train, train_csv, test_csv):
   ftrain = open(train_csv, "w")
   ftest = open(test_csv, "w")
   cnt = 0
   for l in open(src_csv):
       if split_to_train[cnt]:
           ftrain.write(l)
       else:
           ftest.write(l)
       cnt = cnt + 1
   ftrain.close()
   ftest.close()

# Load the list of all the training frames, and shuffle them
# Shuffle the training frames
random.seed(10)
train_frames = get_frames("./data/train")
random.shuffle(train_frames)
validate_frames = get_frames("./data/validate")

# Write the corresponding label information of each frame into file.
write_label_csv("./train-label.csv", train_frames, get_label_map("./data/train.csv"))
write_label_csv("./validate-label.csv", validate_frames, None)

# Dump the data of each frame into a CSV file, apply crop to 64 preprocessor
train_lst = write_data_csv("./train-64x64-data.csv", train_frames, lambda x: crop_resize(x, 64))
valid_lst = write_data_csv("./validate-64x64-data.csv", validate_frames, lambda x: crop_resize(x, 64))

# Generate local train/test split, which you could use to tune your model locally.
train_index = np.loadtxt("./train-label.csv", delimiter=",")[:,0].astype("int")
train_set, test_set = local_split(train_index)
split_to_train = [x in train_set for x in train_index]
split_csv("./train-label.csv", split_to_train, "./local_train-label.csv", "./local_test-label.csv")
split_csv("./train-64x64-data.csv", split_to_train, "./local_train-64x64-data.csv", "./local_test-64x64-data.csv")
