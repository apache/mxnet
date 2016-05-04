import os
import sys
import random
import shlex
import time
import re

from utils import to_bool
from .feat_readers.common import *
from .feat_readers import stats

class DataReadStream(object):

    SCHEMA = {
        "type": "object",
        "properties": {
            "gpu_chunk": {"type": ["string", "integer"], "required": False},

            "lst_file": {"type": "string"},
            "separate_lines": {"type": ["string", "integer", "boolean"], "required": False},
            "has_labels": {"type": ["string", "integer", "boolean"], "required": False},

            "file_format": {"type": "string"},
            "train_stat": {"type": "string", "required": False},
            "offset_labels": {"type": ["string", "integer", "boolean"], "required": False},

            #"XXXchunk": {"type": ["string", "integer"], "required": False},
            "max_feats": {"type": ["string", "integer"], "required": False},
            "shuffle": {"type": ["string", "integer", "boolean"], "required": False},

            "seed": {"type": ["string", "integer"], "required": False},
            "_num_splits": {"type": ["string", "integer"], "required": False},
            "_split_id": {"type": ["string", "integer"], "required": False}
        }
    }

    END_OF_DATA = -1
    END_OF_PARTITION = -2
    END_OF_SEQ = (None, None, None)
    def __init__(self, dataset_args, n_ins):

        # stats
        self.mean = None
        self.std = None
        if 'train_stat' in dataset_args.keys():
            train_stat = dataset_args['train_stat']
            featureStats = stats.FeatureStats()
            featureStats.Load(train_stat)
            self.mean = featureStats.GetMean()
            self.std = featureStats.GetInvStd()

        # open lstfile
        file_path = dataset_args["lst_file"]
        if file_path.endswith('.gz'):
            file_read = gzip.open(file_path, 'r')
        else:
            file_read = open(file_path, 'r')

        separate_lines = False
        if "separate_lines" in dataset_args:
            separate_lines = to_bool(dataset_args["separate_lines"])

        self.has_labels = True
        if "has_labels" in dataset_args:
            self.has_labels = to_bool(dataset_args["has_labels"])

        # parse it, file_lst is a list of (featureFile, labelFile) pairs in the input set
        lines = [ln.strip() for ln in file_read]
        lines = [ln for ln in lines if ln != "" ]

        if self.has_labels:
            if separate_lines:
                if len(lines) % 2 != 0:
                    print("List has mis-matched number of feature files and label files")
                    sys.exit(1)
                self.orig_file_lst = []
                for i in xrange(0, len(lines), 2):
                    self.orig_file_lst.append((lines[i], lines[i+1]))
            else:
                self.orig_file_lst = []
                for i in xrange(len(lines)):
                    pair = re.compile("\s+").split(lines[i])
                    if len(pair) != 2:
                        print(lines[i])
                        print("Each line in the train and eval lists must contain feature file and label file separated by space character")
                        sys.exit(1)
                    self.orig_file_lst.append(pair)
        else:
            # no labels
            self.orig_file_lst = []
            for i in xrange(0, len(lines), 1):
                self.orig_file_lst.append((lines[i], None))

        # save arguments

        self.n_ins = n_ins
        self.file_format = dataset_args['file_format']

        self.file_format = "htk"
        if 'file_format' in dataset_args:
            self.file_format = dataset_args['file_format']

        self.offsetLabels = False
        if 'offset_labels' in dataset_args:
            self.offsetLabels = to_bool(dataset_args['offset_labels'])

        self.chunk_size = 32768
        if 'gpu_chunk' in dataset_args:
            self.chunk_size = int(dataset_args['gpu_chunk'])

        self.maxFeats = 0
        if "max_feats" in dataset_args:
            self.maxFeats = int(dataset_args["max_feats"])
        if self.maxFeats == 0:
            self.maxFeats = sys.maxint

        self.shuffle = True
        if 'shuffle' in dataset_args:
            self.shuffle = to_bool(dataset_args['shuffle'])

        self.seed = None
        if "seed" in dataset_args:
            self.seed = int(dataset_args["seed"])

        if int("_split_id" in dataset_args) + int("_num_splits" in dataset_args) == 1:
            raise Exception("_split_id must be used with _num_splits")
        self.num_splits = 0
        if "_num_splits" in dataset_args:
            self.num_splits = int(dataset_Args["_num_splits"])
            self.split_id = dataset_args["_split_id"]

        # internal state
        self.split_parts = False
        self.by_matrix = False
        self.x = numpy.zeros((self.chunk_size, self.n_ins), dtype=numpy.float32)
        if self.has_labels:
            self.y = numpy.zeros((self.chunk_size,), dtype=numpy.int32)
        else:
            self.y = None
        self.numpy_rng = numpy.random.RandomState(self.seed)

        #self.make_shared()
        self.initialize_read()

    def read_by_part(self):
        if self.file_format in ["kaldi"]:
            self.read_by_matrix()
        else:   # htk
            self.split_parts = True

    def read_by_matrix(self):
        self.by_matrix = True


    def get_shared(self):
        return self.shared_x, self.shared_y

    def initialize_read(self):
        self.file_lst = self.orig_file_lst[:]
        if self.shuffle:
            self.numpy_rng.shuffle(self.file_lst)
        self.fileIndex = 0
        self.totalFrames = 0
        self.reader = None
        self.crossed_part = False
        self.done = False
        self.utt_id = None
        self.queued_feats = None
        self.queued_tgts = None

    def _end_of_data(self):
        return self.totalFrames >= self.maxFeats or self.fileIndex >= len(self.file_lst)

    def _queue_get(self, at_most):
        # if we have frames/labels queued, return at_most of those and queue the rest
        if self.queued_feats is None:
            return None

        num_queued = self.queued_feats.shape[0]
        at_most = min(at_most, num_queued)

        if at_most == num_queued:   # no leftover after the split
            feats, tgts = self.queued_feats, self.queued_tgts
            self.queued_feats = None
            self.queued_tgts = None
        else:
            feats, self.queued_feats = numpy.array_split(self.queued_feats, [at_most])
            if self.queued_tgts is not None:
                tgts, self.queued_tgts = numpy.array_split(self.queued_tgts, [at_most])
            else:
                tgts = None

        return feats, tgts

    def _queue_excess(self, at_most, feats, tgts):
        assert(self.queued_feats is None)
        num_supplied = feats.shape[0]

        if num_supplied > at_most:
            feats, self.queued_feats = numpy.array_split(feats, [at_most])
            if tgts is not None:
                tgts, self.queued_tgts = numpy.array_split(tgts, [at_most])

        return feats, tgts

    # Returns frames/labels (if there are any) or None (otherwise) for current partition
    # Always set the pointers to the next partition
    def _load_fn(self, at_most):
        tup = self._queue_get(at_most)
        if tup is not None:
            return tup

        if self.reader is None:
            featureFile, labelFile = self.file_lst[self.fileIndex]
            self.reader = getReader(self.file_format, featureFile, labelFile)

        if self.reader.IsDone():
            self.fileIndex += 1
            self.reader.Cleanup()
            self.reader = None # cleanup
            return None

        tup = self.reader.Read()
        if tup is None:
            self.fileIndex += 1
            self.reader.Cleanup()
            self.reader = None # cleanup
            return None

        feats, tgts = tup

        # normalize here
        if self.mean is not None:
            feats -= self.mean
        if self.std is not None:
            feats *= self.std

        self.utt_id = self.reader.GetUttId()

        if feats.shape[1] != self.n_ins:
            errMs = "Dimension of features read does not match specified dimensions".format(feats.shape[1], self.n_ins)

        if self.has_labels and tgts is not None:
            if feats.shape[0] != tgts.shape[0]:
                errMs = "Number of frames in feature ({}) and label ({}) files does not match".format(self.featureFile, self.labelFile)
                raise FeatureException(errMsg)

            if self.offsetLabels:
                tgts = numpy.add(tgts, - 1)

        feats, tgts = self._queue_excess(at_most, feats, tgts)

        return feats, tgts

    def current_utt_id(self):
        assert(self.by_matrix or self.split_parts)
        return self.utt_id

    def load_next_seq(self):
        if self.done:
            return DataReadStream.END_OF_SEQ
        if self._end_of_data():
            if self.reader is not None:
                self.reader.Cleanup()
            self.reader = None
            self.done = True
            return DataReadStream.END_OF_SEQ

        num_feats = 0
        old_fileIndes = self.fileIndex

        self.utt_id = None

        tup  = self._load_fn(self.chunk_size)
        if tup is None:
            return DataReadStream.END_OF_SEQ
        (loaded_feats, loaded_tgts) = tup
        return loaded_feats, loaded_tgts, self.utt_id


    def load_next_block(self):
        # if anything left...
        # set_value

        if self.crossed_part:
            self.crossed_part = False
            if not self.by_matrix: #    <--- THERE IS A BUG IN THIS
                return DataReadStream.END_OF_PARTITION
        if self.done:
            return DataReadStream.END_OF_DATA
        if self._end_of_data():
            if self.reader is not None:
                self.reader.Cleanup()
            self.reader = None # cleanup
            self.done = True
            return DataReadStream.END_OF_DATA

        # keep loading features until we pass a partition or EOF

        num_feats = 0
        old_fileIndex = self.fileIndex

        self.utt_id = None

        while num_feats < self.chunk_size:
            if self.split_parts:
                if old_fileIndex != self.fileIndex:
                    self.crossed_part = True
                    break

            if self._end_of_data():
                break

            tup = self._load_fn(self.chunk_size - num_feats)
            if tup is None:
                continue

            (loaded_feat, loaded_label) = tup

            if self.has_labels and loaded_label is None:
                print >> sys.stderr, "Missing labels for: ", self.utt_id
                continue

            numFrames = loaded_feat.shape[0]

            # limit loaded_feat, loaded_label, and numFrames to maximum allowed
            allowed = self.maxFeats - self.totalFrames
            if numFrames > allowed:
                loaded_feat = loaded_feat[0:allowed]
                if self.has_labels:
                    loaded_label = loaded_label[0:allowed]
                numFrames = allowed
                assert(numFrames == loaded_feat.shape[0])

            self.totalFrames += numFrames
            new_num_feats = num_feats + numFrames

            # if the x and y buffers are too small, make bigger ones
            # not possible any more; buffers are always fixed
            """
            if new_num_feats > self.x.shape[0]:
                newx = numpy.zeros((new_num_feats, self.n_ins), dtype=numpy.float32)
                newx[0:num_feats] = self.x[0:num_feats]
                self.x = newx

                if self.has_labels:
                    newy = numpy.zeros((new_num_feats,), dtype=numpy.int32)
                    newy[0:num_feats] = self.y[0:num_feats]
                    self.y = newy
            """

            # place into [num_feats:num_feats+num_loaded]
            self.x[num_feats:new_num_feats] = loaded_feat
            if self.has_labels:
                self.y[num_feats:new_num_feats] = loaded_label

            num_feats = new_num_feats

            if self.by_matrix:
                break

        # if we loaded features, shuffle and copy to shared
        if num_feats != 0:

            if self.shuffle:
                x = self.x[0:num_feats]
                state = self.numpy_rng.get_state()
                self.numpy_rng.shuffle(x)
                self.x[0:num_feats] = x

                if self.has_labels:
                    y = self.y[0:num_feats]
                    self.numpy_rng.set_state(state)
                    self.numpy_rng.shuffle(y)
                    self.y[0:num_feats] = y

            assert(self.x.shape == (self.chunk_size, self.n_ins))
            self.shared_x.set_value(self.x, borrow = True)
            if self.has_labels:
                self.shared_y.set_value(self.y, borrow = True)

            #import hashlib
            #print self.totalFrames, self.x.sum(), hashlib.sha1(self.x.view(numpy.float32)).hexdigest()

            if self.by_matrix:
                self.crossed_part = True

        return num_feats

    def get_state(self):
        return self.numpy_rng.get_state()

    def set_state(self, state):
        self.numpy_rng.set_state(state)
