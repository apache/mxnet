"""
Output dependent functions given a list of function dependnecies

This module searches the given directory or file for functions that are
dependent on the given list of functions. The current directory is used if
none is provided. This script is designed only for python files; it uses
python's ast module to parse python files and find function calls. The
function calls are then compared to the list of dependencies and if there is
a match, the top-level function name is added to the set of dependnet
functions.
"""
import sys
import os
import argparse
import ast
import logging
import re
import itertools 
import json

DEFAULT_CONFIG_FILE = os.path.join(
    os.path.dirname(__file__), "config.json")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def read_config(filename):
    """Reads cross-file dependencies from json file"""
    with open(filename) as f:
        return json.load(f)

def find_dependents(dependencies, top):
    dependents = {}
    top = os.path.abspath(top)

    for filename in dependencies.keys():
        funcs = dependencies[filename]
        abs_path = os.path.join(top, filename)
        deps = set(funcs)
        deps |= find_dependents_file(funcs, abs_path)
        dependents[filename] = deps

    try:
        file_deps = read_config(DEFAULT_CONFIG_FILE)
    except FileNotFoundError:
        file_deps = {}
        logger.WARNING("No config file found, "
            "continuing with no file dependencies")

    for filename in list(dependents.keys()):
        if filename in file_deps:
            for dependent in file_deps[filename]:
                dependents[dependent] = dependents[filename]

    return dependents



def find_dependents_file(dependencies, filename):
    """ Recursively search a file for dependent functions, given dependencies
    """
    class CallVisitor(ast.NodeVisitor):
        def visit_Name(self, node):
            return node.id

        def visit_Attribute(self, node):
            try:
                return "{}.{}".format(node.value.id, node.attr)
            except AttributeError:
                return "{}.{}".format(self.generic_visit(node), node.attr)

    if not dependencies:
        return dependencies
    dependencies = set(dependencies)

    if os.path.splitext(filename)[1] !=".py":
        logger.debug("Skipping non-python file: %s", filename)
        return set()

    with open(filename) as f:
        tree = ast.parse(f.read())
    logger.debug("seaching: %s", filename)

    dependents = set()
    cv = CallVisitor()

    for t in tree.body:     # search for function calls matching dependencies
        if isinstance(t, ast.FunctionDef):
            name = t.name
        else:
            name = "top-level"

        for n in ast.walk(t):
            if isinstance(n, ast.Call):
                func = cv.visit(n.func)
                if func in dependencies:
                    dependents.add(name)

    
    try:
        dependents |= find_dependents_file(dependents - dependencies, filename)
    except RecursionError as re:
        logger.error("Encountered recursion error when seaching {}: {}",
                filename, re.args[0])

    return dependents


def output_results(dependents):
    logger.debug("dependents: %s", dependents)
    for d in dependents:
        print(d)


def parse_args():
    class DependencyAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, "dependencies", {})
            for v in values:
                dep = v.split(":")
                if len(dep) != 2:
                    raise ValueError("Invalid format for dependency " + v +
                                     "Format: <file>:<func-name>.)")
                try:
                    namespace.dependencies[dep[0]].append(dep[1])
                except KeyError:
                    namespace.dependencies[dep[0]] = [dep[1]]

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("dependencies", nargs="+", action=DependencyAction,
        help="list of dependent functions")

    arg_parser.add_argument("--path", "-p", default=".",
        help="directory in which given files are located")

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger.debug("args: %s", args)
    
    dependents = find_dependents(args.dependencies, args.path)
    output_results(dependents)

