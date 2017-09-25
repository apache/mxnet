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

import re
import sys
import operator

def process_output(command_output):
    warnings = {}
    regex = r"(.*):\swarning:\s(.*)"
    lines = command_output.split("\n")
    for line in lines[:-2]:
        matches = re.finditer(regex, line)
        for matchNum, match in enumerate(matches):
            try:
                warnings[match.group()] +=1
            except KeyError:
                warnings[match.group()] =1
    time = lines[-2]
    return time, warnings

def generate_stats(warnings):
    total_count = sum(warnings.values())
    sorted_warnings = sorted(warnings.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_warnings, total_count

def print_summary(time, warnings):
    sorted_warnings, total_count = generate_stats(warnings)
    print "START - Compilation warnings count"
    print total_count, 'warnings'
    print "END - Compilation warnings count"
    print 'START - Compilation warnings summary'
    print 'Time taken to compile:', time, 's'
    print 'Total number of warnings:', total_count, '\n'
    if total_count>0:
        print 'Below is the list of unique warnings and the number of occurrences of that warning'
        for warning, count in sorted_warnings:
            print count, ': ', warning
    print 'END - Compilation warnings summary'

c_output = open(sys.argv[1],'r')
time, warnings = process_output(c_output.read())
print_summary(time, warnings)
