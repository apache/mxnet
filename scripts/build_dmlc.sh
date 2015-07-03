#! /bin/bash

if [ ! -d mshadow ]; then
    git clone https://github.com/dmlc/mshadow.git
fi

if [ ! -d rabit ]; then
    git clone https://github.com/dmlc/rabit.git
fi

if [ ! -d dmlc-core ]; then
    git clone https://github.com/dmlc/dmlc-core.git
fi


if [ ! -f config.mk ]; then
    echo "Use the default config.m"
    cp make/config.mk config.mk
fi

cd rabit
make -j4
cd ..

cd dmlc-core
make -j4
cd ..
