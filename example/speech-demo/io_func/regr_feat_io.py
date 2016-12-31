import os
import sys
import random
import shlex
import time
import re

from utils.utils import to_bool
from feat_readers.common import *
from feat_readers import stats
from feat_io import DataReadStream

class RegrDataReadStream(object):

    def __init__(self, dataset_args, n_ins):
        dataset_args["has_labels"] = False
        assert("seed" in dataset_args)

        args1 = dict(dataset_args)
        args2 = dict(dataset_args)

        args1["lst_file"] = dataset_args["input_lst_file"]
        args2["lst_file"] = dataset_args["output_lst_file"]

        self.input = DataReadStream(args1, n_ins)
        self.output = DataReadStream(args2, n_ins)

    def read_by_part(self):
        self.input.read_by_part()
        self.output.read_by_part()

    def read_by_matrix(self):
        self.input.read_by_matrix()
        self.output.read_by_matrix()

    def make_shared(self):
        self.input.make_shared()
        self.output.make_shared()
        
    def get_shared(self):
        iret = self.input.get_shared()
        oret = self.output.get_shared()
        assert(iret[1] is None)
        assert(oret[1] is None)
        return iret[0], oret[0]

    def initialize_read(self):
        self.input.initialize_read()
        self.output.initialize_read()

    def current_utt_id(self):
        a = self.input.current_utt_id()
        b = self.output.current_utt_id()
        assert(a == b)
        return a

    def load_next_block(self):
        a = self.input.load_next_block()
        b = self.output.load_next_block()        
        assert(a == b)
        return a

    def get_state(self):
        a = self.input.get_state()
        b = self.output.get_state()   
        assert(a[0] == b[0])
        assert(a[2] == b[2])
        assert(a[3] == b[3])
        assert(a[4] == b[4])
        assert(numpy.array_equal(a[1], b[1]))
        return a

    def set_state(self, state):
        self.input.set_state(state)
        self.output.set_state(state)        
