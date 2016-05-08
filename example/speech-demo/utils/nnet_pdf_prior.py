"""
Port of kaldi-head/src/nnet/nnet-pdf-prior.cc
"""

import logging
import sys
import numpy

import theano
import theano.tensor as T

class PdfPrior:

    def __init__(self, class_frame_counts="", prior_scale=1.0, prior_cutoff=1e-10):
        self.logger = logging.getLogger(__name__)
        self.class_frame_counts = class_frame_counts
        self.prior_scale = prior_scale
        self.prior_cutoff = prior_cutoff

        finfo = numpy.finfo('float32')

        self.logger.info("Computing pdf-priors from : %s" % (self.class_frame_counts,))

        text_file = open(self.class_frame_counts, "r")
        content = text_file.read()
        tmp_priors = numpy.array(content.strip().strip('[]').strip().split(' '), dtype='float32')
        text_file.close()

        prior_dim = len(tmp_priors)
        tmp_mask = numpy.zeros(prior_dim)
        num_cutoff = 0
        for i in xrange(prior_dim):
            if tmp_priors[i] < self.prior_cutoff:
                tmp_priors[i] = self.prior_cutoff
                tmp_mask[i] = finfo.max / 2
                num_cutoff += 1

        if num_cutoff > 0:
            self.logger.info("warning: %s out of %s classes have counts lower than %s"
                % (num_cutoff, prior_dim, self.prior_cutoff))

        total = numpy.sum(tmp_priors)
        tmp_priors = tmp_priors / total
        tmp_priors = numpy.log(tmp_priors)
        for i in xrange(prior_dim):
            if not finfo.min < tmp_priors[i] < finfo.max:
                raise Exception()

        tmp_priors_f = tmp_priors[:]
        tmp_priors_f += tmp_mask

        self.log_priors_ = tmp_priors_f
        print self.log_priors_
        self.log_priors = theano.shared(value= self.prior_scale * self.log_priors_, borrow=True)

    # llk is matrix
    def SubtractOnLogpost(self, llk):
        # llk rows += -self.prior_scale * self.log_priors
        return llk - self.log_priors

if __name__ == "__main__":
    #a = PdfPrior(class_frame_counts="/data/sls/scratch/leoliu/experiments/timit/exp1/exp/dnn_512/ali_train_pdf.counts")
    a = PdfPrior(class_frame_counts="/usr/users/leoliu/src/v3-sls-pdnn/configs/sample_frame_counts.txt")
    print a.log_priors_
