# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

from sort_io import BucketSentenceIter, default_build_vocab
from rnn_model import BiLSTMInferenceModel

def MakeInput(char, vocab, arr):
    idx = vocab[char]
    tmp = np.zeros((1,))
    tmp[0] = idx
    arr[:] = tmp

if __name__ == '__main__':
    batch_size = 1
    buckets = []
    num_hidden = 300
    num_embed = 512
    num_lstm_layer = 2

    num_epoch = 1
    learning_rate = 0.1
    momentum = 0.9

    contexts = [mx.context.gpu(i) for i in range(1)]

    vocab = default_build_vocab("./data/sort.train.txt")
    rvocab = {}
    for k, v in vocab.items():
        rvocab[v] = k
    
    _, arg_params, __ = mx.model.load_checkpoint("sort", 1)

    model = BiLSTMInferenceModel(5, len(vocab),
                                 num_hidden=num_hidden, num_embed=num_embed,
                                 num_label=len(vocab), arg_params=arg_params, ctx=contexts, dropout=0.0)

    tks = sys.argv[1:]
    data = np.zeros((1, len(tks)))
    for k in range(len(tks)):
        data[0][k] = vocab[tks[k]]
    
    data = mx.nd.array(data)
    prob = model.forward(data)
    for k in range(len(tks)):        
        print(rvocab[np.argmax(prob, axis = 1)[k]])
    
