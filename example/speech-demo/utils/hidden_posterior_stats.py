"""
x = hidden posterior

1) For each hidden layer

    stat_1:
    pi=1/(1+exp(-x)
    pi=pi/sum(pj)
    sum over each file

    stat_2:
    no sigmoid involved
    pass all the vectors through softmax
    sum all the vectors per file

    stat_3:
    need pi vector concatenated with 1-pi vector
    (result is a bigger vector)

II) For the final layer

    stat_1:
    pass hidden posteriors through softmax
    not the same as the final output of the network (since there is no sigmoid)

Save format for DNN with 2 hidden layers:
${stats_dir}/
    hidden_0/
        basename.stat_1
        basename.stat_2
        basename.stat_3
    hidden_1/
        basename.stat_1
        basename.stat_2
        basename.stat_3
    final/
        basename.stat_1
    hidden_posteriors/  <- only if save_hidden_posteriors == true
        basename.L0
        basename.L1
        basename.L2     <- final layer
        ...

All the .stat_ files (which each store a single vector)
are stored in one big column of numbers instead of 1 row of numbers.
As before, the .L* hidden posterior files still contain one row of numbers per frame.
"""

import os
import logging
import numpy
import sys

import utils

def numpy_sigmoid(X):
    """
    numpy.array -> numpy.array
    compute sigmoid function: 1 / (1 + exp(-X))
    Dunno why this comment is here: All elememnts should be in [0, 1]
    """
    return 1. / (1. + numpy.exp(-X))

def numpy_normalize_rows(a):
    row_sums = a.sum(axis=1)
    new_matrix = a / row_sums[:, numpy.newaxis]
    return new_matrix

def numpy_softmax(X):
    """
    http://nbviewer.ipython.org/github/dolaameng/tutorials/blob/master/ml-tutorials/COURSE_deep_learning.ipynb
    numpy.array -> numpy.array
    Compute softmax function: exp(X) / sum(exp(X, 1))
    where each row of X is a vector output (e.g., different columns representing 
    outputs for different classes)
    The output of softmax is a matrix, with the sum of each row to be nearly 1.0
    as it is the probabilities that are calculated.
    """
    mx = numpy.max(X)
    ex = numpy.exp(X - mx) # prefer zeros over stack overflow - but NOT always useful
    return ex / numpy.sum(ex, 1).reshape(-1, 1)

class FinalStat1:
    def __init__(self, part_posterior_sum=None, part_answer=None):
        self.part_posterior_sum = part_posterior_sum
        self.part_answer = part_answer

class HiddenPosteriorStats:

    def __init__(self, stats_dir, save_hidden_posteriors, num_hidden, n_outs):
        self.logger = logging.getLogger(__name__)

        self.stats_dir = stats_dir
        self.save_hidden_posteriors = save_hidden_posteriors
        self.num_hidden = num_hidden
        self.n_outs = n_outs

        # create directory structure
        if self.stats_dir is not None:
            for i in xrange(self.num_hidden):
                folder = "%s/hidden_%d" %(self.stats_dir, i)
                utils.makedirs(folder)
            folder = "%s/final" %(self.stats_dir,)
            utils.makedirs(folder)

            if save_hidden_posteriors:
                self.logger.info("saving hidden posteriors")
                self.hidden_posteriors_dir = "%s/hidden_posteriors" % (self.stats_dir, )
                utils.makedirs(self.hidden_posteriors_dir)

        # used for asserts
        self.state = 0

        # some stats stuff
        self.total_feats = 0
        self.diction = {str(x):[0,0] for x in xrange(self.n_outs)}

    def partition_start(self, name):
        assert(self.state == 0)
        self.state = 1

        self.part_feats = 0

        self.name = name # save the name for other functions

        if self.stats_dir is not None:
            self.hidden_file1 = []
            self.hidden_file2 = []
            self.hidden_file3 = []
            for i in xrange(self.num_hidden):
                folder = "%s/hidden_%d" %(self.stats_dir, i)
                basename = "%s/%s" % (folder, name)
                self.hidden_file1 += [file(basename+".stat_1", 'w')]
                self.hidden_file2 += [file(basename+".stat_2", 'w')]
                self.hidden_file3 += [file(basename+".stat_3", 'w')]
            folder = "%s/final" %(self.stats_dir,)
            basename = "%s/%s" % (folder, name)
            self.final_file1 = file(basename+".stat_1", 'w')

            if self.save_hidden_posteriors:
                basename = "%s/%s" % (self.hidden_posteriors_dir, name)
                self.f_handle = []
                for i in xrange(self.num_hidden + 1):   # +1 for hidden layer
                    self.f_handle += [file(basename + ".L" + str(i), 'w')]

    def partition_update(self, hidden_posteriors, post, ans):
        """
        for each hidden layer:
            update hidden_stat1[i]
            update hidden_stat2[i]
            update hidden_stat3[i]
        update final_stat1
        append to save posterior dir...
        """
        assert(self.state == 1 or self.state == 2)

        if self.stats_dir is not None:
            hidden_layer_sigmoids = []
            hidden_layer_softmaxs = []
            for i in xrange(self.num_hidden):
                hidden_layer_sigmoids += [numpy_sigmoid(hidden_posteriors[i])]
                hidden_layer_softmaxs += [numpy_softmax(hidden_posteriors[i])]

            hidden_stat1_update = [numpy_normalize_rows(x).sum(axis=0) for x in hidden_layer_sigmoids]
            hidden_stat2_update = [x.sum(axis=0) for x in hidden_layer_softmaxs]
            hidden_stat3_update = [numpy.append(numpy.sum(x, axis=0), numpy.sum(1.-x, axis=0)) for x in hidden_layer_sigmoids]

        # TODO: should we be summing the log posteriors or the posteriors
        # if log posteriors, uncomment 2 lines below
        # post = numpy.log(post)
        # post = numpy.nan_to_num(post)
        
        post = post.sum(axis=0)

        if self.state == 1:
            self.state = 2
            # first update of partition
            if self.stats_dir is not None:
                self.hidden_stat1 = hidden_stat1_update
                self.hidden_stat2 = hidden_stat2_update
                self.hidden_stat3 = hidden_stat3_update
            self.final_stat1 = FinalStat1(part_posterior_sum=post, part_answer=ans[0])
        else:
            # not the first update of partition
            if self.stats_dir is not None:
                for i in xrange(self.num_hidden):
                    self.hidden_stat1[i] += hidden_stat1_update[i]
                    self.hidden_stat2[i] += hidden_stat2_update[i]
                    self.hidden_stat3[i] += hidden_stat3_update[i]
            self.final_stat1.part_posterior_sum += post
            assert(self.final_stat1.part_answer == ans[0])
            
        self.part_feats += len(ans)
        self.total_feats += len(ans)
        
        # append hidden posteriors
        if self.save_hidden_posteriors:
            for i in xrange(len(hidden_posteriors)):
                numpy.savetxt(self.f_handle[i], hidden_posteriors[i])

    def partition_end(self):
        """
        if partition not empty:
            for each hidden layer:
                write hidden_file1[i]
                write hidden_file2[i]
                write hidden_file3[i]
            write final_stat1
            print final_stat1:right/wrong, update diction, print diction
        close all files
        """
        assert(self.state != 0)

        if self.state == 2:
            # write hidden_files
            if self.stats_dir is not None:
                for i in xrange(self.num_hidden):
                    numpy.savetxt(self.hidden_file1[i], self.hidden_stat1[i])
                    numpy.savetxt(self.hidden_file2[i], self.hidden_stat2[i])
                    numpy.savetxt(self.hidden_file3[i], self.hidden_stat3[i])
                # write final_file1
                numpy.savetxt(self.final_file1, self.final_stat1.part_posterior_sum)

            # print final_stat1:right/wrong, update diction, print diction

            wrong = self.final_stat1.part_posterior_sum.argmax() != self.final_stat1.part_answer
            if wrong:
                self.diction[str(self.final_stat1.part_answer)][0] += 1
            self.diction[str(self.final_stat1.part_answer)][1] += 1
    
            out = "total_feats %s feats_in_part %s %s %s " % (self.total_feats, self.part_feats, self.final_stat1.part_posterior_sum, self.final_stat1.part_answer)
            if wrong:
                out += "WRONG"
            else:
                out += "CORRECT"
            self.logger.debug(self.name + " >>>>> " + out)

        # close all files
        if self.stats_dir is not None:
            for i in xrange(self.num_hidden):
                self.hidden_file1[i].close()
                self.hidden_file2[i].close()
                self.hidden_file3[i].close()
            self.final_file1.close()

            if self.save_hidden_posteriors:
                for i in xrange(len(self.f_handle)):
                    if self.f_handle[i] is not None:
                        self.f_handle[i].close()
                    self.f_handle[i] = None

        self.state = 0