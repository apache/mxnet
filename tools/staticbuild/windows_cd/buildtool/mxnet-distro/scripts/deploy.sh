#!/usr/bin/env bash

nosetests -v $(ls mxnet-build/tests/python/unittest/test_gluon*.py | grep -v data | grep -v model_zoo)

wheel_name=$(ls -t dist | head -n 1)
if [[ (! -z $TRAVIS_TAG) || ( $TRAVIS_EVENT_TYPE == 'cron' ) ]]; then
    cp ./.pypirc ~/
    twine upload -r legacy dist/$wheel_name
fi
