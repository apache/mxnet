#! /bin/sh

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


echo "Running the check_regression.sh script"
cat blc_output.txt | uniq | grep -Eo "(http|https).* " | sort| uniq > unique_current_urls.txt

cat url_list.txt unique_current_urls.txt | sort | uniq > new_url_list.txt
regression=false
while IFS= read -r line
do
	err=$(curl -Is $line | head -n 1 | grep 404)
	if [ "$err" ]; then
		if [ "$regression" = false ] ; then
			echo "FAIL: REGRESSION"
			regression=true
		fi
		echo "BROKEN $line $err"
	fi
	unset err
done < new_url_list.txt
mv new_url_list.txt url_list.txt
rm -rf unique_current_urls.txt
rm -rf blc_output.txt
if [ $regression ]; then
	echo "FAIL: Found Regression in broken link checker"
	exit 1
else
	echo "SUCCESS: No Regression found"
fi
