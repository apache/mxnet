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
Output a list of differences between current branch and master

Precondition: this script is run inside an existing git repository

This script first retrieves the raw output from git diff. By default,
the current and master branches are used as targets for git diff,
but the user may specify their own targets. Then, the raw output is 
parsed to retrieve info about each of the changes between the targets, 
including file name, top-level funtion name, and line numbers. 
Finally, the list of changes is outputted.
"""

import os
import subprocess
import sys
import re
import argparse
import logging

logger = logging.getLogger(__name__)


def get_diff_output(args):
    """Perform a git diff using provided args"""
    diff_cmd = ["git", "diff", "--unified=0"]
    if args.commits is not None:
        diff_cmd.extend([args.commits[0], args.commits[1]])
    else:
        if args.branches is None:            
            args.branches = ["master", "HEAD"]
        diff_target = args.branches[0] + "..." + args.branches[1]
        diff_cmd.append(diff_target)

    if args.path:
        diff_cmd.extend(["--", args.path])

    logger.debug("Command: %s", diff_cmd)
    try:
        return subprocess.check_output(diff_cmd)
    except subprocess.CalledProcessError as e:
        logger.error("git diff returned a non zero exit code: %d",
                      e.returncode)
        sys.exit(1)


def parser(diff_output):
    """Split diff output into patches and parse each indiviudally"""
    diff_output = diff_output.decode("utf-8")
    top = subprocess.check_output(["git","rev-parse", "--top-level"])
    top = top.decode("utf-8")
    changes = {}

    for patch in diff_output.split("diff --git")[1:]:
        file_name, cs = parse_patch(patch)
        if not cs:
            continue
        changes[file_name] = cs
    
    return changes


def parse_patch(patch):
    """ Parse changes in a single patch

    Git diff outputs results as patches, each of which corresponds 
    to a single file that has been changed. Each patch consists of 
    a header and one or more hunks that show differing lines between 
    files versions. Hunks themselves have headers, which include 
    line numbers changed and function names.
    """
    lines = patch.splitlines()
    file_name  = lines[0].split()[-1][2:]
    changes = {}
    
    logger.debug("Parsing: %s", file_name)
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
            try:
                hunk_name = tokens[tokens.index("def") + 1].split("(")[0]
            except ValueError:
                hunk_name = "top-level"
            logger.debug("\tHunk: %s - (%d,%d)", hunk_name, start, end)

            # Add hunk info to changes
            if hunk_name not in changes:
                changes[hunk_name] = []
            changes[hunk_name].append((start, end))

        # newly defined top-level function
        if line.startswith("+def "):
            func_name = line.split()[1].split("(")[0]
            changes[func_name] = []
            logger.debug("\tFound new top-level function: %s", func_name)

    return file_name, changes


def output_changes(changes, verbosity=2):
    """ Output changes in an easy to understand format
    
    Three verbosity levels: 
    1 - only file names, 
    2- file and functions names,
    3- file and function names and line numbers.

    Example (verbosity 3):
    file1
        func_a
            1:2
            3:4
        func_b
            5:5
        func_c
    """
    logger.debug("verbosity: %d", verbosity)

    if not changes:
        logger.info("No changes found")
    else:
        for file_name, chunks in changes.items():    
            logger.info(file_name)
            if verbosity < 2:
                continue
            for func_name, ranges in chunks.items():
                logger.info("\t%s", func_name)
                if verbosity < 3:
                    continue
                for (start, end) in ranges:
                    logger.info("\t\t%s:%s", start, end)

    

def parse_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--verbosity", "-v", action="count", 
        help="verbosity level, repeat up to 3 times, defaults to 2")
    arg_parser.add_argument(
        "--logging-level", "-l", dest="level", default="INFO",
        help="logging level, defaults to INFO")

    targets = arg_parser.add_mutually_exclusive_group()
    targets.add_argument(
        "--commits", "-c", nargs=2, metavar=("HASH1 ","HASH2"),
        help="specifies two commits to be compared")
    targets.add_argument(
        "--branches", "-b", nargs=2, metavar=("MASTER", "TOPIC"),
        help="specifies two branches to be compared")

    filters = arg_parser.add_argument_group(
        "filters", "filter which files should be included in output")
    filters.add_argument(
        "--filter-path", "-p", dest="path", 
        help="specify directory or file in which to search for changes")
    filters.add_argument(
        "--filter", "-f", dest="expr", metavar="REGEX", default=".*",
        help="filter files with given python regular expression")
    
    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    try:
        logging.basicConfig(level=getattr(logging, args.level))
    except AttributeError:
        logging.basicConfig(level=logging.INFO)
        logging.warning("Invalid logging level: %s", args.level)
    logging.debug("args: %s", args)

    diff_output = get_diff_output(args)

    changes = parser(diff_output)
    for file_name, chunks in changes.items():
        if not re.match(args.expr, file_name):
            del changes[file_name]

    output_changes(changes, args.verbosity)

