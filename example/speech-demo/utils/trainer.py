import glob
import os
import sys
import logging
import shutil
import math
import StringIO

import utils

import numpy

from io_func.model_io import load, save

"""
Original pdnn setup:

'D:0.08:0.5:0.05,0.05:15'

start_rate 0.08

start decay
  min_derror_decay_start 0.05
  AND
  min_epoch_decay_start 15
  
scale_by 0.5
  
stop decay
  min_derror_stop 0.05

self.learn_rate = 0.1
self.halving_factor = 0.5    
self.max_iters = 20
self.min_iters = 0
self.keep_lr_iters = 15
self.start_halving_impr=0.01
self.end_halving_impr=0.001
"""

def _isnum(n):
    return (not math.isnan(n) and not math.isinf(n))

class Trainer:
    SCHEMA = {}

    def __init__(self, arguments, model, train_fn, valid_fn,
                 train_sets, valid_sets):
        self.logger = logging.getLogger(__name__)

        self.model = model
        self.train_fn = train_fn
        self.valid_fn = valid_fn
        self.train_sets = train_sets
        self.valid_sets = valid_sets

        #################### parse configs #################### 

        self.resume = False
        if "resume" in arguments:
            self.resume = utils.to_bool(arguments["resume"])

        self.wdir = arguments["wdir"]
        self.output_file = arguments["output_file"]

        self.learn_rate = 0.1
        if "learn_rate" in arguments:
            self.learn_rate = float(arguments["learn_rate"])

        self.halving_factor = 0.5    
        if "halving_factor" in arguments:
            self.halving_factor = float(arguments["halving_factor"])
        self.max_iters = 20
        if "max_iters" in arguments:
            self.max_iters = int(arguments["max_iters"])
        self.min_iters = 0
        if "min_iters" in arguments:
            self.min_iters = int(arguments["min_iters"])
        self.keep_lr_iters = 15
        if "keep_lr_iters" in arguments:
            self.keep_lr_iters = int(arguments["keep_lr_iters"])
        self.start_halving_impr=0.01
        if "start_halving_impr" in arguments:
            self.start_halving_impr = float(arguments["start_halving_impr"])
        self.end_halving_impr=0.001
        if "end_halving_impr" in arguments:
            self.end_halving_impr = float(arguments["end_halving_impr"])

        self.continue_with_rate = False
        if "continue_with_rate" in arguments:
            self.continue_with_rate = utils.to_bool(arguments["continue_with_rate"])

        self.halving_criteria = "loss"
        if "halving_criteria" in arguments:
            self.halving_criteria = arguments["halving_criteria"]
        criteria_list = ["loss", "frame_err"]
        if self.halving_criteria not in criteria_list:
            raise Exception("invalid halving criteria. must be one of " + str(criteria_list))

        # batch_size and momentum
        self.batch_size=256
        if arguments.has_key('batch_size'):
            self.batch_size = int(arguments['batch_size'])

        self.momentum=0.5
        self.momentum_start = 1
        if arguments.has_key('momentum'):
            self.momentum = float(arguments['momentum'])
        if 'momentum_start' in arguments:
            self.momentum_start = int(arguments['momentum_start'])

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
        print >>out, "\n********** Trainer **********"
        print >>out, "resume", self.resume
        print >>out, "wdir", self.wdir
        print >>out, "output_file", self.output_file
        print >>out, "learn_rate", self.learn_rate
        print >>out, "halving_factor", self.halving_factor
        print >>out, "max_iters", self.max_iters
        print >>out, "min_iters", self.min_iters
        print >>out, "keep_lr_iters", self.keep_lr_iters
        print >>out, "start_halving_impr", self.start_halving_impr
        print >>out, "end_halving_impr", self.end_halving_impr
        print >>out, "continue_with_rate", self.continue_with_rate
        print >>out, "halving_criteria", self.halving_criteria
        print >>out, "batch_size", self.batch_size
        print >>out, "momentum", self.momentum
        print >>out, "momentum_start", self.momentum_start
        self.logger.info(out.getvalue())

        self.mlp_init = self.wdir + "/mlp_init"
        if not self.resume: # brand new
            save(self.model, self.mlp_init)

        # runtime state
        self.iter = 0
        self.done = False
        self.loss = sys.float_info.max
        self.rate = self.learn_rate
        self.mlp_best = self.mlp_init
        self.halving = False
        self.wasAccepted = True

        if self.resume:
            if os.path.isfile(self.wdir+"/trainer_state"):
                self._load_state()

    def _load_state(self):
        obj = utils.pickle_load(self.wdir+"/trainer_state")
        self.iter = obj["iter"]
        self.done = obj["done"]
        self.loss = obj["loss"]
        self.rate = obj["rate"]
        self.mlp_best = obj["mlp_best"]
        self.halving = obj["halving"]
        self.wasAccepted = obj["wasAccepted"]
        self.train_sets.set_state(obj["train_sets"])
        self.valid_sets.set_state(obj["valid_sets"])

        out = StringIO.StringIO()
        print >>out, "\n********** Resuming from **********"
        print >>out, "iter", self.iter
        print >>out, "done", self.done
        print >>out, "loss", self.loss
        print >>out, "rate", self.rate
        print >>out, "mlp_best", self.mlp_best
        print >>out, "halving", self.halving
        print >>out, "wasAccepted", self.wasAccepted
        self.logger.info(out.getvalue())

    def _save_state(self):
        obj = {}
        obj["iter"] = self.iter
        obj["done"] = self.done
        obj["loss"] = self.loss
        obj["rate"] = self.rate
        obj["mlp_best"] = self.mlp_best
        obj["halving"] = self.halving
        obj["wasAccepted"] = self.wasAccepted
        obj["train_sets"] = self.train_sets.get_state()
        obj["valid_sets"] = self.valid_sets.get_state()

        utils.pickle_save(obj, self.wdir+"/trainer_state")

    def finished(self):
        done = self.done or (self.iter >= self.max_iters)
        self.iter += 1
        return done

    def accepted(self, loss_new, mlp_next):
        # accept or reject new parameters (based on objective function)
        self.loss_prev = self.loss
        if loss_new < self.loss or self.iter <= self.keep_lr_iters:
            self.loss = loss_new
            self.mlp_best = mlp_next
            self.wasAccepted = True
            return True
        else:
            self.wasAccepted = False
            return False

    def _finalize_helper(self, loss_new):
        # no learn-rate halving yet, if keep_lr_iters set accordingly
        if self.iter <= self.keep_lr_iters:
            return

        # stopping criterion
        # bug fix (problem in Kaldi too)
        # if start_halving_impr == 0 and we reject a net
        # then we change self.loss_prev w/o changing self.loss
        # which means rel_impr is 0, instead of a negative number
        # This will half when start_halving_impr = 0.01
        # but not when it equals 0.0 (border case)

        rel_impr = (self.loss_prev - loss_new) / self.loss_prev
        if self.halving and rel_impr < self.end_halving_impr:
            if self.min_iters > self.iter:
                self.logger.info("we were supposed to finish, but we continue as min_iters : " + str(self.min_iters))
                return
            self.logger.info("finished, too small rel. improvement" + str(rel_impr))
            self.done = True

        # start annealing when improvement is low
        self.logger.info("***** rel_impr vs start_halving %f %f" % (rel_impr, self.start_halving_impr))
        if rel_impr < self.start_halving_impr: # and not self.wasAccepted:
            self.halving = True

        keep_rate = self.continue_with_rate and self.wasAccepted
        if not keep_rate:
            if self.halving:
                self.rate = self.rate * self.halving_factor

    def finalize(self, loss_new):
        self._finalize_helper(loss_new)
        self._save_state()

    def _validate(self):
        #print "CVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCVCV"

        errors = []
        losses = []
        total_feats = 0
        self.valid_sets.initialize_read()
        while True:
            num_feats = self.valid_sets.load_next_block()
            if num_feats < 0:
                break
            num_batches = int(math.ceil(float(num_feats) / self.batch_size))
            for batch_index in xrange(num_batches):
                start_idx = batch_index * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_feats)
                batch_feats = end_idx - start_idx

                #print "*******************", start_idx, end_idx
                if False:
                    error, loss = 0, 0
                else:
                    error, loss = self.valid_fn(start_idx, end_idx)
                    assert(_isnum(loss))
                errors.append(error)
                losses.append(loss*batch_feats)

                #print "**************************************"
                #for i in xrange(predictions.shape[0]):
                #    print (predictions[i], answers[i]),
                #print "\n"
                #sys.exit(1)

                if False and batch_index >= 0:
                    print "*** batch_index", batch_index, " >>>>> ", error, loss
                    print "--------------------------------"
                    numpy.set_printoptions(threshold='nan')
                    print self.model.sigmoid_layers[0].W.get_value()[0:5,0:5]
                    print "*"
                    print self.model.sigmoid_layers[0].b.get_value()[0:5]
                    print "*"
                    print self.model.sigmoid_layers[1].W.get_value()[0:5,0:5]
                    print "*"
                    print self.model.sigmoid_layers[1].b.get_value()[0:5]
                    print "--------------------------------"


                #var = raw_input("Please enter something: ")
                #print "you entered", var

            total_feats += num_feats
            self.logger.debug("ITER %02d: valid feats: %d error %4f loss %4f rate %4f" % (self.iter, total_feats, 100*float(numpy.sum(errors))/total_feats, numpy.sum(losses)/total_feats, self.rate))
        return float(numpy.sum(errors))/total_feats, numpy.sum(losses)/total_feats

    def _train(self, learning_rate, momentum):
        #print "TRAINTRAINTRAINTRAINTRAINTRAINTRAINTRAINTRAINTRAINTRAINTRAINTRAINTRAINTRAINTRAIN"
        errors = []
        losses = []
        total_feats = 0
        self.train_sets.initialize_read()
        while True:
            num_feats = self.train_sets.load_next_block()
            if num_feats < 0:
                break
            num_batches = int(math.ceil(float(num_feats) / self.batch_size))

            if False and True:
                print "WARNING TAKE THIS OUT"
                #num_batches = 40    # 40

                print "--------------------------------"
                #numpy.set_printoptions(threshold='nan')
                print self.model.sigmoid_layers[0].W.get_value()[0:5,0:5]
                print "*"
                print self.model.sigmoid_layers[0].b.get_value()[0:5]
                print "*"
                print self.model.sigmoid_layers[1].W.get_value()[0:5,0:5]
                print "*"
                print self.model.sigmoid_layers[1].b.get_value()[0:5]
                print "--------------------------------"
                print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
                print self.model.sigmoid_layers[0].delta_W.get_value()[0:5,0:5]
                print "*"
                print self.model.sigmoid_layers[0].delta_b.get_value()[0:5]
                print "*"
                print self.model.sigmoid_layers[1].delta_W.get_value()[0:5,0:5]
                print "*"
                print self.model.sigmoid_layers[1].delta_b.get_value()[0:5]
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

            for batch_index in xrange(num_batches):
                start_idx = batch_index * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_feats)
                batch_feats = end_idx - start_idx

                rate = learning_rate * float(batch_feats) / self.batch_size
                if False:
                    error, loss = 0, 0
                else:
                    error, loss = self.train_fn(start_idx, end_idx, learning_rate=rate, momentum=momentum)
                    assert(_isnum(loss))
                errors.append(error)
                losses.append(loss*batch_feats)

                #self.logger.debug("%d err=%f loss=%f" % (batch_index, error, loss*batch_feats))
                #if batch_index == 10:
                #   sys.exit(1)

                if False and batch_index >= 0:
                    #print "*** batch_index", batch_index, " >>>>> ", error, loss
                    print "--------------------------------"
                    #numpy.set_printoptions(threshold='nan')
                    print self.model.sigmoid_layers[0].W.get_value()[0:5,0:5]
                    print "*"
                    print self.model.sigmoid_layers[0].b.get_value()[0:5]
                    print "*"
                    print self.model.sigmoid_layers[1].W.get_value()[0:5,0:5]
                    print "*"
                    print self.model.sigmoid_layers[1].b.get_value()[0:5]
                    print "--------------------------------"

            #import sys
            #sys.exit(1)

            total_feats += num_feats
            self.logger.debug("ITER %02d: train feats: %d error %4f loss %4f rate %4f" % (self.iter, total_feats, 100*float(numpy.sum(errors))/total_feats, numpy.sum(losses)/total_feats, self.rate))
        #sys.exit(1)
        #print "ERRORS:", errors
        return float(numpy.sum(errors))/total_feats, numpy.sum(losses)/total_feats

    def train(self):
        if not self.resume:
            # cross-validation on original network
            if True:
                valid_error, loss = self._validate()
                assert(_isnum(loss))
            else:
                valid_error, loss = (9999999,9999999)
            self.logger.info("ITER %02d: CROSSVAL PRERUN ERROR %.4f AVG.LOSS %.4f" % (self.iter, 100*valid_error, loss))
            self.loss = loss

        while not self.finished():      
            load(self.model, filename = self.mlp_best, gradients=self.wasAccepted)

            if self.iter >= self.momentum_start:
                moment = self.momentum
            else:
                moment = 0
            if False:
                tr_error, tr_loss = 0, 0
            else:
                tr_error, tr_loss = self._train(learning_rate=self.rate, momentum=moment)
                assert(_isnum(tr_loss))
            self.logger.info("ITER %02d: TRAIN ERROR %03.4f AVG.LOSS %.4f lrate %.6g" % (self.iter, 100*tr_error, tr_loss, self.rate))

            # cross-validation
            valid_error, valid_loss = self._validate()
            self.logger.info("ITER %02d: CROSSVAL ERROR %.4f AVG.LOSS %.4f lrate %.6f" % (self.iter, 100*valid_error, valid_loss, self.rate))

            mlp_next = "%s/it%02drate%4f_Terr%.4f_Tloss%4f_CVerr%4f_CVloss%4f" % (self.wdir, self.iter, self.rate, 100*tr_error, tr_loss, 100*valid_error, valid_loss)

            save(self.model, filename = mlp_next+".tmp")

            if self.halving_criteria == "loss":
                loss_new = valid_loss
            elif self.halving_criteria == "frame_err":
                loss_new = valid_error
            else:
                raise Exception("bad halving_criteria")

            if self.accepted(loss_new, mlp_next):
                os.rename(mlp_next+".tmp", mlp_next)
                self.logger.info("accepted")
            else:
                os.rename(mlp_next+".tmp", mlp_next+"_rejected")
                self.logger.info("rejected")

            self.finalize(loss_new)

        # select the best network
        if self.mlp_best == self.mlp_init:
            self.logger.critical("Error training neural network...")
            #sys.exit(1)

        output_file = os.path.join(self.wdir, self.output_file)
        shutil.copy2(self.mlp_best, output_file)
        self.logger.info("Succeeded training: " + output_file)