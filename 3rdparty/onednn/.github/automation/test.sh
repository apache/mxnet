#! /bin/bash

#===============================================================================
# Copyright 2019-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --test-kind)
        TEST_KIND="$2"
        ;;
        --build-dir)
        BUILD_DIR="$2"
        ;;
        --report-dir)
        REPORT_DIR="$2"
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
    shift
done


if [ "$(uname)" == "Linux" ]; then
    export OMP_NUM_THREADS="$(grep -c processor /proc/cpuinfo)"
else
    export OMP_NUM_THREADS="$(sysctl -n hw.physicalcpu)"
fi

if [ "${TEST_KIND}" == "gtest" ]; then
    CTEST_OPTS="-E benchdnn"
elif [ "${TEST_KIND}" == "benchdnn" ]; then
    CTEST_OPTS="-R benchdnn --verbose"
else
    echo "Error: unknown test kind: ${TEST_KIND}"
    exit 1
fi

if  [[ ! -z "${REPORT_DIR}" ]]; then
    export GTEST_OUTPUT="${REPORT_DIR}/report/test_report.xml"
fi

CTEST_OPTS="${CTEST_OPTS} --output-on-failure"

echo "CTest options: ${CTEST_OPTS}"
cd "${BUILD_DIR}"
ctest ${CTEST_OPTS}
result=$?

echo "DONE"
exit $result
