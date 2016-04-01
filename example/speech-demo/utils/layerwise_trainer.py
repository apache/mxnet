import glob
import os
import sys
import utils
import logging
import shutil
import StringIO

import utils

import numpy

from io_func.model_io import load, save

class LayerwiseTrainer:
    SCHEMA = {
        "type": "object",
        "properties": {
            "resume": {"type": ["string", "integer", "boolean"], "required": False},
            "wdir": {"type": "string"},
            "output_file": {"type": "string"},
            "max_iters": {"type": ["string", "integer"], "required": False},
            "first_layer_to_train": {"type": ["string", "integer"], "required": False},
            "last_layer_to_train": {"type": ["string", "integer"], "required": False}
        }
    }

    def __init__(self, arguments, model, train_sets):
        self.logger = logging.getLogger(__name__)

        self.model = model
        self.train_sets = train_sets

        #################### parse configs #################### 

        self.resume = False
        if "resume" in arguments:
            self.resume = utils.to_bool(arguments["resume"])

        self.wdir = arguments["wdir"]
        self.output_file = arguments["output_file"]

        self.max_iters = 20
        if "max_iters" in arguments:
            self.max_iters = int(arguments["max_iters"])

        #self.max_iters_without_impr = 3
        #if "max_iters_without_impr" in arguments:
        #    self.max_iters_without_impr = int(arguments["max_iters_without_impr"])

        self.first_layer_to_train = 0
        if "first_layer_to_train" in arguments:
            self.first_layer_to_train = int(arguments["first_layer_to_train"])

        self.last_layer_to_train = model.n_layers - 1	# number hidden layers - 1
        if "last_layer_to_train" in arguments:
            self.last_layer_to_train = int(arguments["last_layer_to_train"])

        # other stuff
        if self.resume:
            if not os.path.exists(self.wdir):
                raise Exception("wdir must exist if resume=True")
        else:
            if not os.path.exists(self.wdir):
                os.makedirs(self.wdir)
            else:
                self.logger.info("Directory already exists...")

        out = StringIO.StringIO()
        print >>out, "\n********** LayerwiseTrainer **********"
        print >>out, "resume", self.resume
        print >>out, "wdir", self.wdir
        print >>out, "output_file", self.output_file
        print >>out, "max_iters", self.max_iters
        print >>out, "first_layer_to_train", self.first_layer_to_train
        print >>out, "last_layer_to_train", self.last_layer_to_train
        self.logger.info(out.getvalue())

        self.mlp_init = self.wdir + "/mlp_init"
        if not self.resume: # brand new
            save(self.model, self.mlp_init)

        # runtime state
        self.layer_index = self.first_layer_to_train
        self.iter = 0
        self.loss = sys.float_info.max
        self.mlp_best = self.mlp_init
        self.mlp_crrnt = self.mlp_init
        self.iters_without_impr = 0

        if self.resume:
            if os.path.isfile(self.wdir+"/layerwisetrainer_state"):
                self._load_state()

    def _load_state(self):
        obj = utils.pickle_load(self.wdir+"/layerwisetrainer_state")
        self.layer_index = obj["layer_index"]
        self.iter = obj["iter"]
        self.loss = obj["loss"]
        self.mlp_best = obj["mlp_best"]
        self.mlp_crrnt = obj["mlp_crrnt"]
        #self.iters_without_impr = obj["iters_without_impr"]
        self.train_sets.set_state(obj["train_sets"])

        out = StringIO.StringIO()
        print >>out, "\n********** Resuming from **********"
        print >>out, "layer_index", self.layer_index
        print >>out, "iter", self.iter
        print >>out, "loss", self.loss
        print >>out, "mlp_best", self.mlp_best
        print >>out, "mlp_crrnt", self.mlp_crrnt
        self.logger.info(out.getvalue())

        load(self.model, self.mlp_crrnt, gradients=True)

    def _save_state(self):
        obj = {}
        obj["layer_index"] = self.layer_index
        obj["iter"] = self.iter
        obj["loss"] = self.loss
        obj["mlp_best"] = self.mlp_best
        obj["mlp_crrnt"] = self.mlp_crrnt
        #obj["iters_without_impr"] = self.iters_without_impr
        obj["train_sets"] = self.train_sets.get_state()

        utils.pickle_save(obj, self.wdir+"/layerwisetrainer_state")

    def train(self, func):
        for layer_index in xrange(self.layer_index, self.last_layer_to_train+1):
            self.loss = sys.float_info.max

            for epoch in xrange(self.iter, self.max_iters):
                filename, loss = func(layer_index, epoch)
                filename = os.path.join(self.wdir, filename)
                save(self.model, filename)
                if loss < self.loss:
                    self.loss = loss
                    self.mlp_best = filename
                self.mlp_crrnt = filename
                
                #self.logger.info("L%d ITER %02d: %s" % (layer_index, epoch, str(info)))
                self.logger.info("L%d ITER %02d: best-of-this-layer: %s" % (layer_index, epoch, self.mlp_best))

                if epoch != self.max_iters - 1:
                    self.layer_index = layer_index
                    self.iter = epoch + 1
                    self._save_state()

            self.iter = 0
            self.layer_index = layer_index + 1
            self._save_state()

        # select the best network
        if self.mlp_best == self.mlp_init:
            self.logger.critical("Error training neural network...")
            #sys.exit(1)

        output_file = os.path.join(self.wdir, self.output_file)
        shutil.copy2(self.mlp_best, output_file)
        self.logger.info("Succeeded training: " + output_file)