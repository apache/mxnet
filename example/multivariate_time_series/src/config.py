import mxnet as mx
import math

#input data parameters
max_training_examples = None  # limit the number of training examples used
# the proportion of training, validation and testing examples
split = [0.6, 0.2, 0.2]
horizon = 3  # how many time steps ahead do we wish to predict?
time_interval = 60 * 60  # seconds between feature values (data defined)

#model hyperparameters
batch_size = 128  # number of examples to pass into the network/use to compute gradient of the loss function
num_epoch = 100  # how many times to backpropogate and update weights
seasonal_period = 24 * 60 * 60  # seconds between important measurements (tune)
# windowsize used to make a prediction
q = max(24 * 7, math.ceil(seasonal_period / time_interval))
# must be smaller than q!!!, size of filters sliding over the input data
filter_list = [6]
num_filter = 50  # number of each filter size
recurrent_state_size = 50  # number of hidden units for each unrolled recurrent layer
# number of hidden units for each unrolled recurrent layer
recurrent_skip_state_size = 20
optimizer = 'Adam'
optimizer_params = {'learning_rate': 0.001,
                    'beta1': 0.9,
                    'beta2': 0.999}
dropout = 0.1  # dropout probability after convolutional/recurrent and autoregressive layers
# choose recurrent cells for the recurrent layer
rcells = [mx.rnn.GRUCell(num_hidden=recurrent_state_size)]
# choose recurrent cells for the recurrent_skip layer
skiprcells = [mx.rnn.GRUCell(
    num_hidden=recurrent_skip_state_size, prefix="skip_")]

#computational parameters
context = mx.cpu()  # train on cpu because maclyfe
