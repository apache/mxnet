#!/bin/bash

# Copyright 2012-2013 Karel Vesely, Daniel Povey
# 	    2015 Yu Zhang
# Apache 2.0

# Begin configuration section.  
nnet= # Optionally pre-select network to use for getting state-likelihoods
feature_transform= # Optionally pre-select feature transform (in front of nnet)
model= # Optionally pre-select transition model
class_frame_counts= # Optionally pre-select class-counts used to compute PDF priors 

stage=0 # stage=1 skips lattice generation
nj=4
cmd=run.pl
max_active=7000 # maximum of active tokens
min_active=200 #minimum of active tokens
max_mem=50000000 # limit the fst-size to 50MB (larger fsts are minimized)
beam=13.0 # GMM:13.0
latbeam=8.0 # GMM:6.0
acwt=0.10 # GMM:0.0833, note: only really affects pruning (scoring is on lattices).
scoring_opts="--min-lmwt 1 --max-lmwt 10"
skip_scoring=false
use_gpu_id=-1 # disable gpu
#parallel_opts="-pe smp 2" # use 2 CPUs (1 DNN-forward, 1 decoder)
parallel_opts= # use 2 CPUs (1 DNN-forward, 1 decoder)
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # The model directory is one level up from decoding directory.
sdata=$data/split$nj;

mxstring=$4

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
  if [ -z $iter ]; then model=$srcdir/final.mdl; 
  else model=$srcdir/$iter.mdl; fi
fi

for f in $model $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "decode_mxnet.sh: no such file $f" && exit 1;
done


# check that files exist
for f in $sdata/1/feats.scp $model $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# PREPARE THE LOG-POSTERIOR COMPUTATION PIPELINE
if [ -z "$class_frame_counts" ]; then
  class_frame_counts=$srcdir/ali_train_pdf.counts
else
  echo "Overriding class_frame_counts by $class_frame_counts"
fi

# Create the feature stream:
feats="scp:$sdata/JOB/feats.scp"
inputfeats="$sdata/JOB/mxnetInput.scp"


if [ -f $sdata/1/feats.scp ]; then
    $cmd JOB=1:$nj $dir/log/make_input.JOB.log \
        echo NO_FEATURE_TRANSFORM scp:$sdata/JOB/feats.scp \> $inputfeats
fi

# Run the decoding in the queue
if [ $stage -le 0 ]; then
  $cmd $parallel_opts JOB=1:$nj $dir/log/decode.JOB.log \
    $mxstring --data_test $inputfeats \| \
    latgen-faster-mapped --min-active=$min_active --max-active=$max_active --max-mem=$max_mem --beam=$beam --lattice-beam=$latbeam \
    --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    $model $graphdir/HCLG.fst ark:- "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
fi

# Run the scoring
if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir || exit 1;
fi

exit 0;
