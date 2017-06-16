import mxnet as mx
from mxnet import foo
from mxnet.foo import nn, rnn

class RNNModel(nn.Layer):
    def __init__(self, mode, vocab_size, num_embed, num_hidden,
                 num_layers, dropout=0.5, tie_weights=False, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed)
            self.rnn = rnn.FusedRNNCell(num_hidden, num_layers, mode=mode,
                                        dropout=dropout, get_next_state=True,
                                        num_input=num_embed)
            if tie_weights:
                self.decoder = nn.Dense(vocab_size, in_units=num_hidden,
                                        params=self.encoder.params)
            else:
                self.decoder = nn.Dense(vocab_size, in_units=num_hidden)

            self.num_hidden = num_hidden

    def forward(self, F, inputs, hidden):
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.rnn.unroll(None, emb, layout='TNC', merge_outputs=True)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
