require("imager")
require("dplyr")
require("readr")
require("mxnet")

source("iterators.R")

######################################################
### Data import and preperation
### First download MNIST train data at Kaggle: 
###   https://www.kaggle.com/c/digit-recognizer/data
######################################################
train <- read_csv('data/train.csv')
train<- data.matrix(train)

train_data <- train[,-1]
train_data <- t(train_data/255*2-1)
train_label <- as.integer(train[,1])

dim(train_data) <- c(28, 28, 1, ncol(train_data))

##################################################
#### Model parameters
##################################################
random_dim<- 96
gen_features<- 96
dis_features<- 32
image_depth = 1
fix_gamma<- T
no_bias<- T
eps<- 1e-5 + 1e-12
batch_size<- 64


##################################################
#### Generator Symbol
##################################################
data = mx.symbol.Variable('data')

gen_rand<- mx.symbol.normal(loc=0, scale=1, shape=c(1, 1, random_dim, batch_size), name="gen_rand")
gen_concat<- mx.symbol.Concat(data = list(data, gen_rand), num.args = 2, name="gen_concat")

g1 = mx.symbol.Deconvolution(gen_concat, name='g1', kernel=c(4,4), num_filter=gen_features*4, no_bias=T)
gbn1 = mx.symbol.BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=eps)
gact1 = mx.symbol.Activation(gbn1, name='gact1', act_type='relu')

g2 = mx.symbol.Deconvolution(gact1, name='g2', kernel=c(3,3), stride=c(2,2), pad=c(1,1), num_filter=gen_features*2, no_bias=no_bias)
gbn2 = mx.symbol.BatchNorm(g2, name='gbn2', fix_gamma=fix_gamma, eps=eps)
gact2 = mx.symbol.Activation(gbn2, name='gact2', act_type='relu')

g3 = mx.symbol.Deconvolution(gact2, name='g3', kernel=c(4,4), stride=c(2,2), pad=c(1,1), num_filter=gen_features, no_bias=no_bias)
gbn3 = mx.symbol.BatchNorm(g3, name='gbn3', fix_gamma=fix_gamma, eps=eps)
gact3 = mx.symbol.Activation(gbn3, name='gact3', act_type='relu')

g4 = mx.symbol.Deconvolution(gact3, name='g4', kernel=c(4,4), stride=c(2,2), pad=c(1,1), num_filter=image_depth, no_bias=no_bias)
G_sym = mx.symbol.Activation(g4, name='G_sym', act_type='tanh')


##################################################
#### Discriminator Symbol
##################################################
data = mx.symbol.Variable('data')
dis_digit = mx.symbol.Variable('digit')
label = mx.symbol.Variable('label')

dis_digit<- mx.symbol.Reshape(data=dis_digit, shape=c(1,1,10,batch_size), name="digit_reshape")
dis_digit<- mx.symbol.broadcast_to(data=dis_digit, shape=c(28,28,10, batch_size), name="digit_broadcast")

data_concat <- mx.symbol.Concat(list(data, dis_digit), num.args = 2, dim = 1, name='dflat_concat')

d1 = mx.symbol.Convolution(data=data_concat, name='d1', kernel=c(3,3), stride=c(1,1), pad=c(0,0), num_filter=24, no_bias=no_bias)
dbn1 = mx.symbol.BatchNorm(d1, name='dbn1', fix_gamma=fix_gamma, eps=eps)
dact1 = mx.symbol.LeakyReLU(dbn1, name='dact1', act_type='elu', slope=0.25)
pool1 <- mx.symbol.Pooling(data=dact1, name="pool1", pool_type="max", kernel=c(2,2), stride=c(2,2), pad=c(0,0))

d2 = mx.symbol.Convolution(pool1, name='d2', kernel=c(3,3), stride=c(2,2), pad=c(0,0), num_filter=32, no_bias=no_bias)
dbn2 = mx.symbol.BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=eps)
dact2 = mx.symbol.LeakyReLU(dbn2, name='dact2', act_type='elu', slope=0.25)

d3 = mx.symbol.Convolution(dact2, name='d3', kernel=c(3,3), stride=c(1,1), pad=c(0,0), num_filter=64, no_bias=no_bias)
dbn3 = mx.symbol.BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=eps)
dact3 = mx.symbol.LeakyReLU(dbn3, name='dact3', act_type='elu', slope=0.25)

d4 = mx.symbol.Convolution(dact2, name='d3', kernel=c(4,4), stride=c(1,1), pad=c(0,0), num_filter=64, no_bias=no_bias)
dbn4 = mx.symbol.BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=eps)
dact4 = mx.symbol.LeakyReLU(dbn4, name='dact4', act_type='elu', slope=0.25)

# pool4 <- mx.symbol.Pooling(data=dact3, name="pool4", pool_type="avg", kernel=c(4,4), stride=c(1,1), pad=c(0,0))

dflat = mx.symbol.Flatten(dact4, name="dflat")

dfc <- mx.symbol.FullyConnected(data=dflat, name="dfc", num_hidden=1, no_bias=F)
D_sym = mx.symbol.LogisticRegressionOutput(data=dfc, label=label, name='D_sym')


########################
### Graph
########################
input_shape_G<- c(1, 1, 10, batch_size)
input_shape_D<- c(28, 28, 1, batch_size)

graph.viz(G_sym, type = "graph", direction = "LR")
graph.viz(D_sym, type = "graph", direction = "LR")

