#!/bin/bash
# pack mxnet into a single directory

if [ $# -lt 2 ]; then
    echo "usage: $0 language dest_dir"
    echo "  language : python, ..."
    exit
fi

cur_dir=`pwd`
mxnet_dir=$cur_dir/`dirname $0`/../
cd $mxnet_dir; mxnet_dir=`pwd`; cd $cur_dir

lang=$1
dest_dir=`pwd`/$2
mkdir -p $dest_dir

if [ "$lang" == "python" ] ; then
    dest_dir=$dest_dir
    echo "packing python from [$mxnet_dir] to [$dest_dir]"

    rm -rf $dest_dir
    mkdir $dest_dir

    if [ ! -f $mxnet_dir/lib/libmxnet.so ]; then
        echo "didn't find libmxnet.so, run make first"
        exit
    fi

    cp -r $mxnet_dir/python/mxnet $dest_dir
    cp $mxnet_dir/lib/libmxnet.so $dest_dir

    if [ -d $mxnet_dir/deps/lib ]; then
        cp $mxnet_dir/deps/lib/libprotobuf.so $dest_dir
        cp $mxnet_dir/deps/lib/libzmq.so $dest_dir
    fi
    echo "done. then you can set the following environment"
    echo "export LD_LIBRARY_PATH=${dest_dir}:\${LD_LIBRARY_PATH}"
    echo "export PYTHONPATH=${dest_dir}:\${PYTHONPATH}"
else
    echo "unsupported language: $lang"
fi
