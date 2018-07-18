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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def find_dependents(dependencies, top, depth=2):
    """Search an entire directory for dependencies

    A depth of 2 is used as a default, since it is rare for modules
    to cross-reference each other.
    """
    if depth == 0:
        return dependencies

    if os.path.isfile(top):
        return find_dependents_file(set(dependencies), top)
    
    dependents = set()
    for root, dirs, files in os.walk(top):
        for f in files:
            if os.path.splitext(f)[1] == ".py":
                path = os.path.join(root, f)
                dependents.update(find_dependents_file(
                    set(dependencies), path))

    dependents.update(find_dependents(dependents, top, depth-1))
    return dependents


def find_dependents_file(dependencies, filename):
    """ Recursively search a file for dependent functions, given dependencies
    """
    class CallVisitor(ast.NodeVisitor):
        def visit_Name(self, node):
            return node.id

        def visit_Attribute(self, node):
            return node.attr
            #try:
            #    return "{}.{}".format(node.value.id, node.attr)
            #except AttributeError:
            #    return "{}.{}".format(self.generic_visit(node), node.attr)

    if not dependencies:
        return dependencies

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
        loggeg.error("Encountered recursion error when seaching {}: {}",
                filename, re.args[0])

    return dependents


def output_results(dependents):
    logger.debug("dependents: %s", dependents)
    for d in dependents:
        print(d)


def parse_args():
    class DependencyAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, "dependencies", [])
            for v in values:
                dep = re.split("\.py:|\.", v)
                if len(dep) != 2:
                    q
                    raise ValueError("Invalid format for dependency " + v +
                                     "Format: <file-name>.<func-name> or "
                                     "<directory>/<file>:<func-name>.)")
                namespace.dependencies.append((dep[0], dep[1]))

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("dependencies", nargs="+",
        help="list of dependent functions")
    
    arg_parser.add_argument("--path", "-p", default=".",
        help="file or directory in which to search for dependents")

    #arg_parser.add_argument("--")

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger.debug("args: %s", args)
    
    path = os.path.join(os.getcwd(), args.path)
    dependents = find_dependents(args.dependencies, path)
    output_results(dependents)

