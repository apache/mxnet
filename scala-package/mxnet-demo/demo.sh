#!/bin/bash
CURR_DIR=$(cd $(dirname $0); pwd)
CLASSPATH=$CLASSPATH:$CURR_DIR/target/*:$CLASSPATH:$CURR_DIR/target/classes/lib/*
java -Xmx8G  -cp $CLASSPATH sample.HelloWorld
