import numpy as np
import os
import sys
import logging

from StringIO import StringIO
import json


from datetime import datetime

from kaldi_parser import *
import utils.utils as utils

# nicer interface for file2nnet, nnet2file

def load(model, filename, gradients, num_hidden_layers=-1, with_final=True, factors=None):
    _file2nnet(model.sigmoid_layers, set_layer_num = num_hidden_layers,
        filename=filename, activation="sigmoid", withfinal=with_final, factor=1.0, gradients=gradients, factors=factors)

def save(model, filename):
    _nnet2file(model.sigmoid_layers, set_layer_num = -1, filename=filename,
        activation="sigmoid", start_layer = 0, withfinal=True)

# convert an array to a string
def array_2_string(array):
    return array.astype('float32')

# convert a string to an array
def string_2_array(string):
    if isinstance(string, str) or isinstance(string, unicode):
        str_in = StringIO(string)
        return np.loadtxt(str_in)
    else:
        return string

def _nnet2file(layers, set_layer_num = -1, filename='nnet.out', activation='sigmoid', start_layer = 0, withfinal=True, input_factor = 0.0, factor=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]):
    logger = logging.getLogger(__name__)
    logger.info("Saving network "+filename)

    n_layers = len(layers)
    nnet_dict = {}
    if set_layer_num == -1:
        set_layer_num = n_layers - 1

    for i in range(start_layer, set_layer_num):
        logger.info("Saving hidden layer "+str(i))
        dict_a = str(i) + ' ' + activation + ' W'
        if i == 0:
            nnet_dict[dict_a] = array_2_string((1.0 - input_factor) * layers[i].params[0].get_value())
        else:
            nnet_dict[dict_a] = array_2_string((1.0 - factor[i-1]) * layers[i].params[0].get_value())
        dict_a = str(i) + ' ' + activation + ' b'
        nnet_dict[dict_a] = array_2_string(layers[i].params[1].get_value())

        # gradients
        dict_a = str(i) + ' ' + activation + ' dW'
        nnet_dict[dict_a] = array_2_string(layers[i].delta_params[0].get_value())
        dict_a = str(i) + ' ' + activation + ' db'
        nnet_dict[dict_a] = array_2_string(layers[i].delta_params[1].get_value())
    
        if layers[i].kahan:
            logger.info("Loading hidden kahan")
            dict_a = str(i) + ' ' + activation + ' W_carry'
            nnet_dict[dict_a] = array_2_string(layers[i].params_carry[0].get_value())
            dict_a = str(i) + ' ' + activation + ' b_carry'
            nnet_dict[dict_a] = array_2_string(layers[i].params_carry[1].get_value())
            #dict_a = str(i) + ' ' + activation + ' dW_carry'
            #nnet_dict[dict_a] = array_2_string(layers[i].delta_params_carry[0].get_value())
            #dict_a = str(i) + ' ' + activation + ' db_carry'
            #nnet_dict[dict_a] = array_2_string(layers[i].delta_params_carry[1].get_value())

    if withfinal: 
        logger.info("Saving final layer ")
        
        dict_a = 'logreg W'
        nnet_dict[dict_a] = array_2_string((1.0 - factor[-1]) * layers[-1].params[0].get_value())
        dict_a = 'logreg b'
        nnet_dict[dict_a] = array_2_string(layers[-1].params[1].get_value())

        #gradients
        dict_a = 'logreg dW'
        nnet_dict[dict_a] = array_2_string(layers[-1].delta_params[0].get_value())
        dict_a = 'logreg db'
        nnet_dict[dict_a] = array_2_string(layers[-1].delta_params[1].get_value())

        if layers[-1].kahan:
            logger.info("Loading softmax kahan")
            dict_a = 'logreg W_carry'
            nnet_dict[dict_a] = array_2_string(layers[-1].params_carry[0].get_value())
            dict_a = 'logreg b_carry'
            nnet_dict[dict_a] = array_2_string(layers[-1].params_carry[1].get_value())
            #dict_a = 'logreg dW_carry'
            #nnet_dict[dict_a] = array_2_string(layers[-1].delta_params_carry[0].get_value())
            #dict_a = 'logreg db_carry'
            #nnet_dict[dict_a] = array_2_string(layers[-1].delta_params_carry[1].get_value())

    utils.pickle_save(nnet_dict, filename)   

def zero(x):
    x.set_value(np.zeros_like(x.get_value(borrow=True), dtype=theano.config.floatX))

def _file2nnet(layers, set_layer_num = -1, filename='nnet.in', activation='sigmoid', withfinal=True, factor=1.0, gradients=False, factors=None):
    logger = logging.getLogger(__name__)
    logger.info("Loading "+filename)

    # if is KALDI binary
    if fileIsBinary(filename):
        print "Warning dropout factors ignored here"

        nnet = file2nnet_binary(filename)

        n_layers = len(nnet)
        if set_layer_num == -1:
            set_layer_num = n_layers - 1

        for i in xrange(set_layer_num):
            layers[i].params[0].set_value(factor * nnet[i]["weights"].astype(dtype=theano.config.floatX))
            layers[i].params[1].set_value(nnet[i]["bias"].astype(dtype=theano.config.floatX))

        if withfinal:
            #print nnet[-1]["weights"][0][0:10]
            layers[-1].params[0].set_value(nnet[-1]["weights"].astype(dtype=theano.config.floatX))
            layers[-1].params[1].set_value(nnet[-1]["bias"].astype(dtype=theano.config.floatX))

        return

    # else, it's pdnn format

    n_layers = len(layers)

    if factors is None:
        factors = [1.0 for l in layers]

    if len(factors) != n_layers:
        raise Exception("number of factors does not equal number of hidden + softmax")

    nnet_dict = {}
    if set_layer_num == -1:
        set_layer_num = n_layers - 1

    nnet_dict = utils.pickle_load(filename)

    for i in xrange(set_layer_num):
        logger.info("Loading hidden layer "+str(i))

        dict_key = str(i) + ' ' + activation + ' W'
        layers[i].params[0].set_value(factors[i] * factor * np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))
        dict_key = str(i) + ' ' + activation + ' b' 
        layers[i].params[1].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))

        if gradients:
            dict_key = str(i) + ' ' + activation + ' dW'
            layers[i].delta_params[0].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))
            dict_key = str(i) + ' ' + activation + ' db' 
            layers[i].delta_params[1].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))            
        else:
            zero(layers[i].delta_params[0])
            zero(layers[i].delta_params[1])

        dict_key = str(i) + ' ' + activation + ' W_carry'
        if layers[i].kahan and dict_key in nnet_dict:
            logger.info("Loading hidden kahan")
            dict_key = str(i) + ' ' + activation + ' W_carry'
            layers[i].params_carry[0].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))
            dict_key = str(i) + ' ' + activation + ' b_carry' 
            layers[i].params_carry[1].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))            
            #dict_key = str(i) + ' ' + activation + ' dW_carry'
            #layers[i].delta_params_carry[0].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))
            #dict_key = str(i) + ' ' + activation + ' db_carry' 
            #layers[i].delta_params_carry[1].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))            

        if layers[i].sync:
            layers[i].params_sync[0].set_value(layers[i].params[0].get_value().astype('float32'))
            layers[i].params_sync[1].set_value(layers[i].params[1].get_value().astype('float32'))
            logger.info("Copy params to sync")

    if withfinal:
        logger.info("Loading final layer ")

        dict_key = 'logreg W'
        layers[-1].params[0].set_value(factors[-1] * np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))
        dict_key = 'logreg b'
        layers[-1].params[1].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))
        if gradients:
            dict_key = 'logreg dW'
            layers[-1].delta_params[0].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))
            dict_key = 'logreg db'
            layers[-1].delta_params[1].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))
        else:
            zero(layers[-1].delta_params[0])
            zero(layers[-1].delta_params[1])

        dict_key = 'logreg W_carry'
        if layers[-1].kahan and dict_key in nnet_dict:
            logger.info("Loading softmax kahan")
            dict_key = 'logreg W_carry'
            layers[-1].params_carry[0].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))
            dict_key = 'logreg b_carry' 
            layers[-1].params_carry[1].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))            
            #dict_key = 'logreg dW_carry'
            #layers[-1].delta_params_carry[0].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))
            #dict_key = 'logreg db_carry' 
            #layers[-1].delta_params_carry[1].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))            

        if layers[-1].sync:
            layers[-1].params_sync[0].set_value(layers[-1].params[0].get_value().astype('float32'))
            layers[-1].params_sync[1].set_value(layers[-1].params[1].get_value().astype('float32'))
            logger.info("Copy softmax params to sync")

    if gradients:
        logger.info("Loading gradients")
    else:
        logger.info("Zero-ing gradients")

def _cnn2file(conv_layers, filename='nnet.out', activation='sigmoid', withfinal=True, input_factor = 1.0, factor=1.0):
    n_layers = len(conv_layers)
    nnet_dict = {}
    for i in xrange(n_layers):
       conv_layer = conv_layers[i]
       filter_shape = conv_layer.filter_shape
       
       for next_X in xrange(filter_shape[0]):
           for this_X in xrange(filter_shape[1]):
               dict_a = 'W ' + str(i) + ' ' + str(next_X) + ' ' + str(this_X) 
               if i == 0:
                   nnet_dict[dict_a] = array_2_string(input_factor * (conv_layer.W.get_value())[next_X, this_X])
               else:
                   nnet_dict[dict_a] = array_2_string(factor * (conv_layer.W.get_value())[next_X, this_X])

       dict_a = 'b ' + str(i)
       nnet_dict[dict_a] = array_2_string(conv_layer.b.get_value())
    
    with open(filename, 'wb') as fp:
        json.dump(nnet_dict, fp, indent=2, sort_keys = True)
        fp.flush()

def _file2cnn(conv_layers, filename='nnet.in', activation='sigmoid', withfinal=True, factor=1.0):
    n_layers = len(conv_layers)
    nnet_dict = {}

    with open(filename, 'rb') as fp:
        nnet_dict = json.load(fp)
    for i in xrange(n_layers):
        conv_layer = conv_layers[i]
        filter_shape = conv_layer.filter_shape
        W_array = conv_layer.W.get_value()

        for next_X in xrange(filter_shape[0]):
            for this_X in xrange(filter_shape[1]):
                dict_a = 'W ' + str(i) + ' ' + str(next_X) + ' ' + str(this_X)
                W_array[next_X, this_X, :, :] = factor * np.asarray(string_2_array(nnet_dict[dict_a]))

        conv_layer.W.set_value(W_array) 

        dict_a = 'b ' + str(i)
        conv_layer.b.set_value(np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX)) 
