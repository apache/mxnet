#!/usr/bin/env bash
# Build and push all docker containers

DEVICES=('cpu' 'gpu')
LANGUAGES=('python' 'julia' 'r-lang' 'scala' 'perl')
for DEV in "${DEVICES[@]}"; do
    for LANG in "${LANGUAGES[@]}"; do
        ./tool.sh build ${LANG} ${DEV}
        ./tool.sh push ${LANG} ${DEV}
    done
done
