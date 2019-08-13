# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""This example demonstrates how the LSTM-CRF model can be implemented
in Gluon to perform noun-phrase chunking as a sequence labeling task.
"""
import sys
import mxnet as mx
from mxnet import autograd as ag, ndarray as nd, gluon
from mxnet.gluon import Block, nn, rnn
import mxnet.optimizer as optim

mx.random.seed(1)


# Helper functions to make the code more readable.
def to_scalar(x):
    return int(x.asscalar())


def argmax(vec):
    # return the argmax as a python int
    idx = nd.argmax(vec, axis=1)
    return to_scalar(idx)


def prepare_sequence(seq, word2Idx):
    return nd.array([word2Idx[w] for w in seq])


# Compute log sum exp is numerically more stable than multiplying probabilities
def log_sum_exp(vec):
    max_score = nd.max(vec).asscalar()
    return nd.log(nd.sum(nd.exp(vec - max_score))) + max_score


# Model
class BiLSTM_CRF(Block):
    """Get BiLSTM_CRF model"""
    def __init__(self, vocab_size, tag2Idx, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        with self.name_scope():
            self.embedding_dim = embedding_dim
            self.hidden_dim = hidden_dim
            self.vocab_size = vocab_size
            self.tag2idx = tag2Idx
            self.tagset_size = len(tag2Idx)
            self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = rnn.LSTM(hidden_dim // 2, num_layers=1, bidirectional=True)

            # Maps the output of the LSTM into tag space.
            self.hidden2tag = nn.Dense(self.tagset_size)

            # Matrix of transition parameters.  Entry i,j is the score of
            # transitioning *to* i *from* j.
            self.transitions = self.params.get("crf_transition_matrix", shape=(self.tagset_size, self.tagset_size))
            self.hidden = self.init_hidden()

    def init_hidden(self):
        return [nd.random.normal(shape=(2, 1, self.hidden_dim // 2)),
                nd.random.normal(shape=(2, 1, self.hidden_dim // 2))]

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        alphas = [[-10000.] * self.tagset_size]
        alphas[0][self.tag2idx[START_TAG]] = 0.
        alphas = nd.array(alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].reshape((1, -1))
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions.data()[next_tag].reshape((1, -1))
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = alphas + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            alphas = nd.concat(*alphas_t, dim=0).reshape((1, -1))
        terminal_var = alphas + self.transitions.data()[self.tag2idx[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentences):
        self.hidden = self.init_hidden()
        length = sentences.shape[0]
        embeds = self.word_embeds(sentences).reshape((length, 1, -1))
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.reshape((length, self.hidden_dim))
        lstm_feats = self.hidden2tag(lstm_out)
        return nd.split(lstm_feats, num_outputs=length, axis=0, squeeze_axis=True)

    def _score_sentence(self, feats, tags_array):
        # Gives the score of a provided tag sequence
        score = nd.array([0])
        tags_array = nd.concat(nd.array([self.tag2idx[START_TAG]]), *tags_array, dim=0)
        for idx, feat in enumerate(feats):
            score = score + \
                    self.transitions.data()[to_scalar(tags_array[idx+1]),
                                            to_scalar(tags_array[idx])] + feat[to_scalar(tags_array[idx+1])]
        score = score + self.transitions.data()[self.tag2idx[STOP_TAG],
                                                to_scalar(tags_array[int(tags_array.shape[0]-1)])]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        vvars = nd.full((1, self.tagset_size), -10000.)
        vvars[0, self.tag2idx[START_TAG]] = 0

        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = vvars + self.transitions.data()[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0, best_tag_id])
            # Now add in the emission scores, and assign vvars to the set
            # of viterbi variables we just computed
            vvars = (nd.concat(*viterbivars_t, dim=0) + feat).reshape((1, -1))
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = vvars + self.transitions.data()[self.tag2idx[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0, best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag2idx[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentences, tags_list):
        feats = self._get_lstm_features(sentences)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags_list)
        return forward_score - gold_score

    def forward(self, sentences):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentences)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


# Run training
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word2idx = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

tag2idx = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word2idx), tag2idx, EMBEDDING_DIM, HIDDEN_DIM)
model.initialize(mx.init.Xavier(magnitude=2.24), ctx=mx.cpu())
optimizer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd': 1e-4})

# Check predictions before training
precheck_sent = prepare_sequence(training_data[0][0], word2idx)
precheck_tags = nd.array([tag2idx[t] for t in training_data[0][1]])
print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data

    neg_log_likelihood_acc = 0.
    iter = 0
    for i, (sentence, tags) in enumerate(training_data):
        # Step 1. Get our inputs ready for the network, that is,
        # turn them into Variables of word indices.
        # Remember to use autograd to record the calculation.
        with ag.record():
            sentence_in = prepare_sequence(sentence, word2idx)
            targets = nd.array([tag2idx[t] for t in tags])

            # Step 2. Run our forward pass.
            neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)

            # Step 3. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            neg_log_likelihood.backward()
        optimizer.step(1)
        neg_log_likelihood_acc += neg_log_likelihood.mean()
        iter = i
    print("Epoch [{}], Negative Log Likelihood {:.4f}".format(epoch, neg_log_likelihood_acc.asscalar()/(iter+1)))

# Check predictions after training
precheck_sent = prepare_sequence(training_data[0][0], word2idx)
print(model(precheck_sent))

# Acknowledgement: this example is adopted from pytorch nlp tutorials.
