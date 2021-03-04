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

echo "Using clang-format version: $(clang-format --version)"
echo "Starting format check..."

for filename in $(find "$(pwd)" -type f | grep -P ".*\.(c|cpp|h|hpp|cl)$"); do clang-format -style=file -i $filename; done

RETURN_CODE=0
echo $(git status) | grep "nothing to commit" > /dev/null

if [ $? -eq 1 ]; then
    echo "Clang-format check FAILED! Found not formatted files!"
    echo "$(git status)"
    RETURN_CODE=3
else
    echo "Clang-format check PASSED! Not formatted files not found..."
fi

exit ${RETURN_CODE}