import logging
import os
import subprocess
import json

import flakiness_checker
import diff_collator
import test_selector

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
logging.getLogger().addHandler(sh)

DEFAULT_CONFIG_FILE = "tools/flaky_test_bot/config.json"
DEFAULT_TESTS_DIRECTORY = "tests/python"

def read_config(filename):
    """Reads cross-file tests dependencies from json file"""
    with open(filename) as f:
        file_deps = json.load(f)

    return file_deps


def select_tests(changes):
    """returns tests that are dependent on given changes
    """
    tests = {}

    top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    top = str(top, errors="strict").splitlines()[0]

    for filename in changes.keys():
        funcs = changes[filename]
        abs_path = os.path.join(top, filename)
        if not os.path.exists(abs_path):
            continue
        deps = set(funcs)
        deps |= test_selector.find_dependents_file(funcs, abs_path)
        tests[filename] = deps

    file_deps = read_config(os.path.join(top, DEFAULT_CONFIG_FILE))
    for filename in list(tests.keys()):
        if filename in file_deps:
            for dependent in file_deps[filename]:
                tests[dependent] = tests[filename]

    for filename, funcs in list(tests.items()):
        tests[filename] = [t for t in funcs if t.startswith("test_")]
        if not tests[filename]:
            del tests[filename]

    return tests

if __name__ == "__main__":
    args = diff_collator.parse_args()
    diff_output = diff_collator.get_diff_output(args)
    changes = diff_collator.parser(diff_output)
    diff_collator.output_changes(changes)

    changes = {k:set(v.keys()) for k, v in  changes.items()}
    tests = select_tests(changes)
    logger.debug("tests:")
    for t, fs in tests.items():
        logger.debug(t)
        for f in fs:
            logger.debug("\t%s", f)

    flaky = []
    for filename, funcs in tests.items():
        for func in funcs:
            res = flakiness_checker.run_test_trials(filename, func, 1)
            if res != 0:
                flaky.append((filename,func))

    if flaky:
        logger.info("tests failed flakiness checker:")
        for t in flaky:
            logger.info(t)

