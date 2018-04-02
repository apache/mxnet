---
layout: post
title: Deep Learning for Multivariate Time Series Forecasting using Apache MXNet
---

This tutorial shows how to implement LSTNet, a multivariate time series forecasting model submitted by Wei-Cheng Chang, Yiming Yang, Hanxiao Liu and Guokun Lai in their paper [Modeling Long- and Short-Term Temporal Patterns](https://arxiv.org/pdf/1703.07015.pdf) in March 2017.  This model achieved state of the art performance on 3 of the 4 public datasets it was evaluated on.

We will use MXNet to train a neural network with convolutional, recurrent, recurrent-skip and autoregressive components.  The result is a model that predicts the future value for all input variables, given a specific horizon.

![](/example/multivariate_time_series/docs/model_architecture.png)

> Image from [Modeling Long- and Short-Term Temporal Patterns
with Deep Neural Networks](https://arxiv.org/abs/1703.07015), Figure 2

Our first step will be to download and unpack the public electricity dataset used in the paper.  This dataset comprises measurements of electricity consumption in kWh every hour from 2012 to 2014 for 321 different clients.

```s
$ wget https://github.com/laiguokun/multivariate-time-series-data/raw/master/electricity/electricity.txt.gz
$ gunzip electricity.txt.gz
```

Now we need to preprocess the data.  Each training record is the previous q values of each time series (see figure above).  Each label is the value of each of the 321 time series, h steps ahead.

```python
#modules
import math
import os
import sys
import numpy as np
import math
import mxnet as mx
import pandas as pd

#custom modules
import config

##############################################
#load input data
##############################################

#read tar.gz data into a pandas dataframe
df = pd.read_csv("./electricity.txt", sep=",", header=None)

#convert to numpy array
x = df.astype(float).as_matrix()

print("\n\tlength of time series: ", x.shape[0])

################################################
# define model hyperparameters and preprocessing parameters
################################################

#input data parameters
max_training_examples = None  # limit the number of training examples used
split = [0.6, 0.2, 0.2] # the proportion of training, validation and testing examples
horizon = 3  # how many time steps ahead do we wish to predict?
time_interval = 60 * 60  # seconds between feature values (data defined)

#model hyperparameters
batch_size = 128  # number of examples to pass into the network/use to compute gradient of the loss function
num_epoch = 100  # how many times to backpropogate and update weights
seasonal_period = 24 * 60 * 60  # seconds between important measurements (tune)
q = max(24 * 7, math.ceil(seasonal_period / time_interval)) # windowsize used to make a prediction
filter_list = [6] #size of each filter we wish to apply
num_filter = 50  # number of each filter size
recurrent_state_size = 50  # number of hidden units for each unrolled recurrent layer
recurrent_skip_state_size = 20 # number of hidden units for each unrolled recurrent-skip layer
optimizer = 'Adam'
optimizer_params = {'learning_rate': 0.001,
                    'beta1': 0.9,
                    'beta2': 0.999}
dropout = 0.1  # dropout probability applied after convolutional/recurrent and autoregressive layers
rcells = [mx.rnn.GRUCell(num_hidden=recurrent_state_size)] #recurrent cell types we wish to use
skiprcells = [mx.rnn.GRUCell(num_hidden=recurrent_skip_state_size, prefix="skip_")] # recurrent-skip cell types we wish to use

#computational parameters
context = mx.cpu()  # train on cpu or gpu


##############################################
# loop through data constructing features/labels
##############################################

#create arrays for storing values in
x_ts = np.zeros((x.shape[0] - q,  q, x.shape[1]))
y_ts = np.zeros((x.shape[0] - q, x.shape[1]))

#loop through collecting records for ts analysis depending on q
for n in list(range(x.shape[0])):

    if n + 1 < q:
        continue

    if n + 1 + horizon > x.shape[0]:
        continue

    else:
        y_n = x[n+horizon,:]
        x_n = x[n+1 - q:n+1,:]

    x_ts[n - q] = x_n
    y_ts[n - q] = y_n

#split into training and testing data
training_examples = int(x_ts.shape[0] * split[0])
valid_examples = int(x_ts.shape[0] * split[1])

x_train = x_ts[:training_examples]
y_train = y_ts[:training_examples]
x_valid = x_ts[training_examples:training_examples + valid_examples]
y_valid = y_ts[training_examples:training_examples + valid_examples]
x_test = x_ts[training_examples + valid_examples:]
y_test = y_ts[training_examples + valid_examples:]

print("\ntraining examples: ", x_train.shape[0],
        "\n\nvalidation examples: ", x_valid.shape[0],
        "\n\ntest examples: ", x_test.shape[0],
        "\n\nwindow size: ", q,
        "\n\nskip length p: ", seasonal_period / time_interval)
```

Now that we have our input data we are ready to start building the network graph.   First lets consider the data iterators and convolutional component.  If you're not solid on convolutions check out [this blog post](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/).

All filters have width = number of time series.  The input data is zero padded on one side before being passed into the convolutional layer, ensuring the output from each kernel has the same dimensions, regardless of the filter size.

Each filter slides over the input data producing a 1D array of length q.  Relu activation is used as per the paper.  The resulting output from the convolutional component is of shape (batch size, q, total number of filters).

```python
###############################################
#define input data iterators for training and testing
###############################################

train_iter = mx.io.NDArrayIter(data={'seq_data': x_train},
                               label={'seq_label': y_train},
                               batch_size=batch_size)

val_iter = mx.io.NDArrayIter(data={'seq_data': x_valid},
                             label={'seq_label': y_valid},
                             batch_size=batch_size)

test_iter = mx.io.NDArrayIter(data={'seq_data': x_test},
                             label={'seq_label': y_test},
                             batch_size=batch_size)

#print input shapes
input_feature_shape = train_iter.provide_data[0][1]
input_label_shape = train_iter.provide_label[0][1]
print("\nfeature input shape: ", input_feature_shape,
      "\nlabel input shape: ", input_label_shape)

####################################
# define neural network graph
####################################

#create placeholders to refer to when creating network graph (names are defined in data iterators)
seq_data = mx.symbol.Variable(train_iter.provide_data[0].name)
seq_label = mx.sym.Variable(train_iter.provide_label[0].name)

# reshape data before applying convolutional layer (takes 4D shape)
conv_input = mx.sym.Reshape(data=seq_data, shape=(batch_size, 1, q, x.shape[1]))


print("\n\t#################################\n\
       #convolutional component:\n\
       #################################\n")

#create many convolutional filters to slide over the input
outputs = []
for i, filter_size in enumerate(filter_list):

        # zero pad the input array, adding rows at the top only
        # this ensures the number output rows = number input rows after applying kernel
        padi = mx.sym.pad(data=conv_input, mode="constant", constant_value=0,
                            pad_width=(0, 0, 0, 0, filter_size - 1, 0, 0, 0))                  
        padi_shape = padi.infer_shape(seq_data=input_feature_shape)[1][0]

        # apply convolutional layer (the result of each kernel position is a single number)
        convi = mx.sym.Convolution(data=padi, kernel=(filter_size, x.shape[1]), num_filter=num_filter)
        convi_shape = convi.infer_shape(seq_data=input_feature_shape)[1][0]

        #apply relu activation function as per paper
        acti = mx.sym.Activation(data=convi, act_type='relu')

        #transpose output shape in preparation for recurrent layer (batches, q, num filter, 1)
        transposed_convi = mx.symbol.transpose(data=acti, axes= (0,2,1,3))
        transposed_convi_shape = transposed_convi.infer_shape(seq_data=input_feature_shape)[1][0]

        #reshape to (batches, q, num filter) for recurrent layers
        reshaped_transposed_convi = mx.sym.Reshape(data=transposed_convi, target_shape=(batch_size, q, num_filter))
        reshaped_transposed_convi_shape = reshaped_transposed_convi.infer_shape(seq_data=input_feature_shape)[1][0]

        #append resulting symbol to a list
        outputs.append(reshaped_transposed_convi)

        print("\n\tpadded input size: ", padi_shape)
        print("\n\t\tfilter size: ", (filter_size, x.shape[1]), " , number of filters: ", num_filter)
        print("\n\tconvi output layer shape (notice length is maintained): ", convi_shape)
        print("\n\tconvi output layer after transposing: ", transposed_convi_shape)
        print("\n\tconvi output layer after reshaping: ", reshaped_transposed_convi_shape)

#concatenate symbols to (batch, total_filters, q, 1)
conv_concat = mx.sym.Concat(*outputs, dim=2)
conv_concat_shape = conv_concat.infer_shape(seq_data=input_feature_shape)[1][0]
print("\nconcat output layer shape: ", conv_concat_shape)

#apply a dropout layer
conv_dropout = mx.sym.Dropout(conv_concat, p = dropout)
```

The output from the convolutional layer is used in two places.  The first is a simple recurrent layer.  A gated recurrent unit is unrolled through q time steps.  The output of the last time step is taken.  Here's an [awesome blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) explaining different types of recurrent cells.

```python
print("\n\t#################################\n\
       #recurrent component:\n\
       #################################\n")

#define a gated recurrent unit cell, which we can unroll into many symbols based on our desired time dependancy
cell_outputs = []
for i, recurrent_cell in enumerate(rcells):

    #unroll the lstm cell, obtaining a symbol each time step
    outputs, states = recurrent_cell.unroll(length=conv_concat_shape[1], inputs=conv_dropout, merge_outputs=False, layout="NTC")

    #we only take the output from the recurrent layer at time t
    output = outputs[-1]

    #just ensures we can have multiple RNN layer types
    cell_outputs.append(output)

    print("\n\teach of the ", conv_concat_shape[1], " unrolled hidden cells in the RNN is of shape: ",
          output.infer_shape(seq_data=input_feature_shape)[1][0],
          "\nNOTE: only the output from the unrolled cell at time t is used...")


#concatenate output from each type of recurrent cell
rnn_component = mx.sym.concat(*cell_outputs, dim=1)
print("\nshape after combining RNN cell types: ", rnn_component.infer_shape(seq_data=input_feature_shape)[1][0])

#apply a dropout layer to output
rnn_dropout = mx.sym.Dropout(rnn_component, p=dropout)
```

The output from the convolutional layer is also passed to the recurrent-skip component.  Again a gated recurrent unit is unrolled through q time steps.  Unrolled units a prespecified time interval (seasonal period) apart are connected. In practice recurrent cells do not capture long term dependencies.  When predicting electricity consumption the measurements from the previous day could be very useful predictors.  By introducing skip connections 24 hours apart we ensure the model can leverage these historical dependencies.

```python
print("\n\t#################################\n\
       #recurrent-skip component:\n\
       #################################\n")

#define number of cells to skip through to get a certain time interval back from current hidden state
p =int(seasonal_period / time_interval)
print("adding skip connections for cells ", p, " intervals apart...")

#define a gated recurrent unit cell, which we can unroll into many symbols based on our desired time dependancy
skipcell_outputs = []
for i, recurrent_cell in enumerate(skiprcells):

    #unroll the rnn cell, obtaining an output and state symbol each time
    outputs, states = recurrent_cell.unroll(length=conv_concat_shape[1], inputs=conv_dropout, merge_outputs=False, layout="NTC")

    #for each unrolled timestep
    counter = 0
    connected_outputs = []
    for i, current_cell in enumerate(reversed(outputs)):

        #try adding a concatenated skip connection
        try:

            #get seasonal cell p steps apart
            skip_cell = outputs[i + p]

            #connect this cell to is seasonal neighbour
            cell_pair = [current_cell, skip_cell]
            concatenated_pair = mx.sym.concat(*cell_pair, dim=1)

            #prepend symbol to a list
            connected_outputs.insert(0, concatenated_pair)

            counter += 1

        except IndexError:

            #prepend symbol to a list without skip connection
            connected_outputs.insert(0, current_cell)

    selected_outputs = []
    for i, current_cell in enumerate(connected_outputs):

        t = i + 1

        #use p hidden states of Recurrent-skip component from time stamp t − p + 1 to t as outputs
        if t > conv_concat_shape[1] - p:

            selected_outputs.append(current_cell)

    #concatenate outputs
    concatenated_output = mx.sym.concat(*selected_outputs, dim=1)

    #append to list
    skipcell_outputs.append(concatenated_output)

print("\n\t", len(selected_outputs), " hidden cells used in output of shape: ",
        concatenated_pair.infer_shape(seq_data=input_feature_shape)[1][0], " after adding skip connections")

print("\n\tconcatenated output shape for each skipRNN cell type: ",
      concatenated_output.infer_shape(seq_data=input_feature_shape)[1][0])

#concatenate output from each type of recurrent cell
skiprnn_component = mx.sym.concat(*skipcell_outputs, dim=1)
print("\ncombined flattened recurrent-skip shape : ", skiprnn_component.infer_shape(seq_data=input_feature_shape)[1][0])

#apply a dropout layer
skiprnn_dropout = mx.sym.Dropout(skiprnn_component, p=dropout)
```

The final component is a simple autoregressive layer.  This splits the input data into 321 individual time series and passes each to a fully connected layer of size 1, with no activation function.  The effect of this is to predict the next value as a linear combination of the previous q values.

```python
print("\n\t#################################\n\
       #autoregressive component:\n\
       #################################\n")

auto_list = []
for i in list(range(x.shape[1])):

    #get a symbol representing data in each individual time series
    time_series = mx.sym.slice_axis(data=seq_data, axis=2, begin=i, end=i + 1)

    #pass to a fully connected layer
    fc_ts = mx.sym.FullyConnected(data=time_series, num_hidden=1)

    auto_list.append(fc_ts)

print("\neach time series shape: ", time_series.infer_shape(seq_data=input_feature_shape)[1][0])

#concatenate fully connected outputs
ar_output = mx.sym.concat(*auto_list, dim=1)
print("\nar component shape: ", ar_output.infer_shape(
    seq_data=input_feature_shape)[1][0])

#do not apply activation function since we want this to be linear
```

Now lets combine all the components, define a loss function and create a trainable module from the final symbol.  I found I achieved good performance with the L2 loss function.

```python
print("\n\t#################################\n\
       #combine AR and NN components:\n\
       #################################\n")

#combine model components
neural_components = mx.sym.concat(*[rnn_dropout, skiprnn_dropout], dim=1)

#pass to fully connected layer to map to a single value
neural_output = mx.sym.FullyConnected(
    data=neural_components, num_hidden=x.shape[1])
print("\nNN output shape : ", neural_output.infer_shape(
    seq_data=input_feature_shape)[1][0])

#sum the output from AR and deep learning
model_output = neural_output + ar_output
print("\nshape after adding autoregressive output: ",
      model_output.infer_shape(seq_data=input_feature_shape)[1][0])

#########################################
# loss function
#########################################

#compute the gradient of the L2 loss
loss_grad = mx.sym.LinearRegressionOutput(data=model_output, label=seq_label)

#set network point to back so name is interpretable
batmans_NN = loss_grad

#########################################
# create a trainable module on CPU/GPUs
#########################################

model = mx.mod.Module(symbol=batmans_NN,
                      context=context,
                      data_names=[v.name for v in train_iter.provide_data],
                      label_names=[v.name for v in train_iter.provide_label])
```

We are ready to start training, however, before we do so lets create some custom metrics.  Please see the paper for a definition of the three metrics: Relative square error, relative absolute error and correlation.

Note: although MXNet has functions for creating custom metrics, I found the metric output of my implementation varied with batch size, so defined them explicity.

```python
####################################
#define evaluation metrics to show when training
#####################################

#root relative squared error
def rse(label, pred):
    """computes the root relative squared error
    (condensed using standard deviation formula)"""

    #compute the root of the sum of the squared error
    numerator = np.sqrt(np.mean(np.square(label - pred), axis=None))
    #numerator = np.sqrt(np.sum(np.square(label - pred), axis=None))

    #compute the RMSE if we were to simply predict the average of the previous values
    denominator = np.std(label, axis=None)
    #denominator = np.sqrt(np.sum(np.square(label - np.mean(label, axis = None)), axis=None))

    return numerator / denominator

#relative absolute error
def rae(label, pred):
    """computes the relative absolute error
    (condensed using standard deviation formula)"""

    #compute the root of the sum of the squared error
    numerator = np.mean(np.abs(label - pred), axis=None)
    #numerator = np.sum(np.abs(label - pred), axis = None)

    #compute AE if we were to simply predict the average of the previous values
    denominator = np.mean(np.abs(label - np.mean(label, axis=None)), axis=None)
    #denominator = np.sum(np.abs(label - np.mean(label, axis = None)), axis=None)

    return numerator / denominator

#empirical correlation coefficient
def corr(label, pred):
    """computes the empirical correlation coefficient"""

    #compute the root of the sum of the squared error
    numerator1 = label - np.mean(label, axis=0)
    numerator2 = pred - np.mean(pred, axis=0)
    numerator = np.mean(numerator1 * numerator2, axis=0)

    #compute the root of the sum of the squared error if we were to simply predict the average of the previous values
    denominator = np.std(label, axis=0) * np.std(pred, axis=0)

    #value passed here should be 321 numbers
    return np.mean(numerator / denominator)

#create a composite metric manually
def metrics(label, pred):
    return ["RSE: ", rse(label, pred), "RAE: ", rae(label, pred), "CORR: ", corr(label, pred)]
```

Time to train.

```python
################
# #fit the model
################

# allocate memory given the input data and label shapes
model.bind(data_shapes=train_iter.provide_data,
           label_shapes=train_iter.provide_label)

# initialize parameters by uniform random numbers
model.init_params()

# optimizer
model.init_optimizer(optimizer=optimizer,
                     optimizer_params=optimizer_params)

# train n epochs, i.e. going over the data iter one pass
for epoch in range(num_epoch):

    train_iter.reset()
    val_iter.reset()

    for batch in train_iter:
        # compute predictions
        model.forward(batch, is_train=True)
        model.backward()                                # compute gradients
        model.update()                                  # update parameters

    # compute train metrics
    pred = model.predict(train_iter).asnumpy()
    label = y_train

    print('\n', 'Epoch %d, Training %s' % (epoch, metrics(label, pred)))

    # compute test metrics
    pred = model.predict(val_iter).asnumpy()
    label = y_valid
    print('Epoch %d, Validation %s' % (epoch, metrics(label, pred)))
```

The hyperparameters previously specified resulted in comparible performance to the results in the paper (*RSE = 0.0967, RAE = 0.0581 and CORR = 0.8941*) with horizon = 3 hours.

This model took ~10 hours to train on an [Nvidia Tesla K80 GPU](http://www.nvidia.ca/object/tesla-k80.html).

This code can be found in [my github repo](https://github.com/opringle/multivariate_time_series_forecasting), separated into training and config files. You can find the trained model symbol and parameters in the results folder.  This model was originally implemented in PyTorch and can be found [here](https://github.com/laiguokun/LSTNet).

Happy forecasting!

# About the author

>[Oliver Pringle](https://www.linkedin.com/in/oliverpringle/) graduated from the UBC Master of Data Science Program in 2017 and is currently a Data Scientist at [Finn.ai](http://finn.ai/) working on AI driven conversational assistants.


