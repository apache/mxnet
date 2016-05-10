#!/bin/bash

# This script trains and evaluate LSTM models. There is no
# discriminative training yet.
# In this recipe, MXNet directly read Kaldi features and labels,
# which makes the whole pipline much simpler.

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will
                 #return non-zero return code
set -u           #Fail on an undefined variable

. ./cmd.sh
. ./path.sh

cmd=hostd3.pl
# root folder,
expdir=exp_mxnet

##################################################
# Kaldi generated folder
##################################################

# alignment folder
ali_src=exp_cntk/sdm1/dnn_120fbank_ali

# decoding graph
graph_src=exp/sdm1/tri3a/graph_ami_fsh.o3g.kn.pr1-7/

# features
train_src=data/sdm1/train_fbank_gcmvn
dev_src=data/sdm1/eval_fbank_gcmvn

# config file
config=ami_local_bptt.cfg

# optional settings,
njdec=128
scoring="--min-lmwt 5 --max-lmwt 19"

# The device number to run the training
# change to AUTO to select the card automatically
deviceNumber=gpu1

# decoding method
method=simple
modelName=
# model
prefix=
num_epoch=
acwt=0.1
#smbr training variables
num_utts_per_iter=40
smooth_factor=0.1
use_one_sil=true

stage=0
. utils/parse_options.sh || exit 1;


###############################################
# Training
###############################################

mkdir -p $expdir
dir=$expdir/data-for-mxnet

# prepare listing data
if [ $stage -le 0 ] ; then
    mkdir -p $dir
    mkdir -p $dir/log
    mkdir -p $dir/rawpost

    # for compressed ali
    #$cmd JOB=1:$njdec $dir/log/gen_post.JOB.log \
    #    ali-to-pdf $ali_src/final.mdl "ark:gunzip -c $ali_src/ali.JOB.gz |" \
    #        ark:- | ali-to-post ark:- ark,scp:$dir/rawpost/post.JOB.ark,$dir/rawpost/post.JOB.scp || exit 1;
    num=`cat $ali_src/num_jobs`
    $cmd JOB=1:$num $dir/log/gen_post.JOB.log \
        ali-to-pdf $ali_src/final.mdl ark:$ali_src/ali.JOB.ark \
            ark:- \| ali-to-post ark:- ark,scp:$dir/rawpost/post.JOB.ark,$dir/rawpost/post.JOB.scp || exit 1;


    for n in $(seq $njdec); do
        cat $dir/rawpost/post.${n}.scp || exit 1;
    done > $dir/post.scp
fi

if [ $stage -le 1 ] ; then
    # split the data : 90% train and 10% held-out
    [ ! -e ${train_src}_tr90 ] && utils/subset_data_dir_tr_cv.sh $train_src ${train_src}_tr90 ${train_src}_cv10

    # generate dataset list
    echo NO_FEATURE_TRANSFORM scp:${train_src}_tr90/feats.scp > $dir/train.feats
    echo scp:$dir/post.scp >> $dir/train.feats

    echo NO_FEATURE_TRANSFORM scp:${train_src}_cv10/feats.scp > $dir/dev.feats
    echo scp:$dir/post.scp >> $dir/dev.feats

    echo NO_FEATURE_TRANSFORM scp:${dev_src}/feats.scp > $dir/test.feats
fi

# generate label counts
if [ $stage -le 2 ] ; then
    $cmd JOB=1:1 $dir/log/gen_label_mean.JOB.log \
        python make_stats.py --configfile $config --data_train $dir/train.feats \| copy-feats ark:- ark:$dir/label_mean.ark
    echo NO_FEATURE_TRANSFORM ark:$dir/label_mean.ark > $dir/label_mean.feats
fi


# training, note that weight decay is for the whole batch (0.00001 * 20 (minibatch) * 40 (batch_size))
if [ $stage -le 3 ] ; then
    python train_lstm_proj.py --configfile $config --data_train $dir/train.feats --data_dev $dir/dev.feats --train_prefix $PWD/$expdir/$prefix --train_optimizer speechSGD --train_learning_rate 1 --train_context $deviceNumber --train_weight_decay 0.008 --train_show_every 1000
fi

# decoding
if [ $stage -le 4 ] ; then
  cp $ali_src/final.mdl $expdir
  mxnet_string="OMP_NUM_THREADS=1 python decode_mxnet.py --config $config --data_test $dir/test.feats --data_label_mean $dir/label_mean.feats --train_method $method --train_prefix $PWD/$expdir/$prefix --train_num_epoch $num_epoch --train_context cpu0 --train_batch_size 1"
  ./decode_mxnet.sh --nj $njdec --cmd $decode_cmd --acwt $acwt --scoring-opts "$scoring" \
    $graph_src $dev_src $expdir/decode_${prefix}_$(basename $dev_src) "$mxnet_string" || exit 1;

fi
