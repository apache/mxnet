#!/bin/bash
CURR_DIR=$(cd $(dirname $0); pwd)
cp log4j.properties target/classes/
CLASSPATH=$CLASSPATH:$CURR_DIR/target/*:$CLASSPATH:$CURR_DIR/target/classes/lib/*
java -Xmx8G  -cp $CLASSPATH sample.HelloWorld
