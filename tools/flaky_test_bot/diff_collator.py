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

Precondition: this script is run inside an existing git repository

This script first retrieves the raw output from git diff. By default,
the current and master branches are used as targets for git diff, but the user
may specify their own targets. Then, the raw output is parsed to retrieve info
about each of the changes bnetween the targets, including file name, top-level
funtion name, and line numbers for each change. Finally, the list of changes
is outputted with each change on a separate line.
"""

import os
import subprocess
import sys
import re
import argparse
import logging

logging.basicConfig(level=logging.INFO) 


def get_diff_output(args):
    try:
        if args.commits is not None:
            diff_output = subprocess.check_output(
                ["git", "diff", "--unified=0", args.commits[0],
                args.commits[1], "--", args.path])
        else:
            if args.branches is None:
                # Default to current branch with master
                args.branches = ["master", "HEAD"]

            diff_target = args.branches[0] + "..." + args.branches[1]
            diff_output = subprocess.check_output(
                ["git", "diff", "--unified=0", diff_target, "--", args.path])
    except subprocess.CalledProcessError as e:
        logging.error("git diff returned a non zero exit code: %d",
                      e.returncode)
        sys.exit(1)

    return diff_output


def parser(diff_output):
    """Split diff output into patches and parse each patch individually"""
    diff_output = str(diff_output, errors="strict")
    changes = []

    for patch in diff_output.split("diff --git")[1:]:
        # split diff_output into patches
        changes.append(parse_patch(patch))
    
    return changes


def parse_patch(patch):
    """ Parse changes in a single patch

    Git diff outputs results as patches, each of which corresponds to a single
    that has been changed.  Each patch consists of a header with one or more
    hunks that show differing lines between files.  Hunks themselves have
    headers that include line numbers changed and function names.
    """
    lines = patch.splitlines()

    changes = {}
    file_name  = lines[0].split()[-1][2:]
    
    logging.debug("Parsing: %s", file_name)

    for line in patch.splitlines():
        # parse hunk header
        if line.startswith("@"):
            tokens = line.split()
            to_range = []
            start = 0
            end = 0
            
            # Get line numbers
            for t in tokens[1:]:
                if t.startswith("@"):
                    start = int(to_range[0])
                    try:
                        end = start + int(to_range[1])
                    except IndexError:
                        end = start
                else:
                    to_range = t[1:].split(",")

            # Get function name
            hunk_header = tokens[-1].split("(")
            if len(hunk_header) == 1:
                hunk_name = "top-level"
            else:
                hunk_name = tokens[-1].split("(")[0]
            logging.debug("\tHunk: %s - (%d,%d)", hunk_name, start, end)

            # Add hunk info to changes
            if hunk_name not in changes:
                changes[hunk_name] = []
            changes[hunk_name].append((start, end))

    return (file_name, changes)


def output_changes(changes, verbosity):
    if not verbosity:
        verbosity = 2
    logging.debug("verbosity: %d", verbosity)

    if not changes:
        logging.info("No changes found")
    else:
        for file_name, chunks in changes:
            if verbosity == 1:
                print(file_name)
            for func_name, ranges in chunks.items():
                if verbosity == 2:
                    print("{}\t{}".format(file_name, func_name))
                for (start, end) in ranges:
                    if verbosity > 2:
                        print("{}\t{}\t{}:{}".format(
                            file_name, func_name, start, end))

    

def parse_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--verbosity", "-v", action="count", 
        help="verbosity level, repeat up to 3 times, defaults to 2")

    targets = arg_parser.add_mutually_exclusive_group()
    targets.add_argument(
        "--commits", "-c", nargs=2, metavar=("HASH1 ","HASH2"),
        help="specifies two commits to be compared")
    targets.add_argument(
        "--branches", "-b", nargs=2, metavar=("MASTER", "TOPIC"),
        help="specifies two branches to be compared")

    filters = arg_parser.add_argument_group(
        "filters", "filter which files should be included in output")
    filters.add_argument("--filter-path", "-p", dest="path", default="",
        help="specify directory or file in which to search for changes")
    filters.add_argument(
        "--filter", "-f", "-e", dest="expr", metavar="REGEX", default=".*",
        help="filter files with given python regular expression")
    
    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    diff_output = get_diff_output(args)

    changes = parser(diff_output)
    changes = [(n, cs) for (n, cs) in changes if re.fullmatch(args.expr, n)]
    output_changes(changes, args.verbosity)

