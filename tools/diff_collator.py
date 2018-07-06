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
        changes +=  parse_file(file_diff)

    return changes

def parse_file(file_diff):
    changes = []
    lines = file_diff.splitlines()
    file_name  = lines[0].split()[-1][2:]

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

            changes.append((file_name, func_name, (start, end)))

    return changes

def output_changes(changes):
    for c in changes:
        print(c)
            
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
        git_diff_output = subprocess.check_output(["git", "diff",
            "--unified=0", args.commits[0], args.commits[1]])
    elif args.branches is not None:
        diff_target = args.branches[0] + "..." + args.branches[1]
        git_diff_output = subprocess.check_output(["git", "diff",
            "--unified=0", diff_target])
    else:
        git_diff_output = subprocess.check_output(["git", "diff", "--unified=0"])

    changes = parser(git_diff_output)
    output_changes(changes)

