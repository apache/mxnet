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

"""
Output a list of differences between current git branch and master

Precondition: this script is run instide an existing git repository

This script will performs a retrieves and output a list of changes
based on the output of git diff. For each changes, the file, line numbers
and top-level funtcion name is provided.
"""

import os
import subprocess
import sys
import re
import argparse
import logging

logging.basicConfig(level=logging.INFO) 

def parser(diff_output):
    diff_output = str(diff_output, errors="strict")
    changes = []

    for file_diff in diff_output.split("diff --git")[1:]:
        changes.append(parse_file(file_diff))

    return changes

def parse_file(file_diff):
    """ Parse changes to a single file
    git diff 
    """
    changes = {}
    lines = file_diff.splitlines()
    file_name  = lines[0].split()[-1][2:]
    logging.info("Parsing: %s", file_name)

    for line in file_diff.splitlines():
        if line.startswith("@"):
            # parse hunk header
            tokens = line.split()
            to_range = []
            start = 0
            end = 0
            for t in tokens[1:]:
                if t.startswith("@"):
                    start = int(to_range[0])
                    try:
                        end = start + int(to_range[1])
                    except IndexError:
                        end = start
                else:
                    to_range = t[1:].split(",")

            func_header = tokens[-1].split("(")
            if len(func_header) == 1:
                func_name = "top-level"
            else:
                func_name = tokens[-1].split("(")[0]

            if func_name not in changes:
                changes[func_name] = []
            changes[func_name].append((start, end))

    return (file_name, changes)

def output_changes(changes):
    for file_name, chunks in changes:
        print(file_name)
        for func_name, ranges in chunks.items():
            print("\t{}".format(func_name))
            for (start, end) in ranges:
                print("\t\t{} {}".format(start, end))
    
def parse_args():
    arg_parser = argparse.ArgumentParser()

    group = arg_parser.add_mutually_exclusive_group()
    group.add_argument("--commits", "-c", nargs=2,
                       help="specifies two commits to be compared")
    group.add_argument("--branches", "-b", nargs=2,
                       metavar=["master", "topic"],
                       help="specifies two branches to be compared")

    args = arg_parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.commits is  not None:
        diff_output = subprocess.check_output(["git", "diff",
            "--unified=0", args.commits[0], args.commits[1]])
    elif args.branches is not None:
        diff_target = args.branches[0] + "..." + args.branches[1]
        diff_output = subprocess.check_output(["git", "diff",
                                              "--unified=0", diff_target])
    else:
        diff_output = subprocess.check_output(["git", "diff", "--unified=0",
                                               "master...HEAD"])

    changes = parser(diff_output)
    output_changes(changes)

