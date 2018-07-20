import logging
import os
import subprocess
import json

import flakiness_checker
import diff_collator
import dependency_analyzer

DEFAULT_DIR = os.path.dirname(__file__)
DEFAULT_CONFIG_FILE = os.path.join(DEFAULT_DIR, "config.json")
DEFAULT_LOGGING_FILE = os.path.join(DEFAULT_DIR, "results.log")
DEFAULT_TESTS_DIRECTORY = "tests/python"
DEFAULT_NUM_TRIALS = 5000

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(DEFAULT_LOGGING_FILE)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
logger.addHandler(fh)

logging.getLogger().setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(
    "[%(levelname)s] %(name)s: %(message)s"))
sh.setLevel(logging.DEBUG)
logging.getLogger().addHandler(sh)


def select_tests(changes):
    """returns tests that are dependent on given changes
    """

    top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    top = str(top, errors="strict").splitlines()[0]

    tests = dependency_analyzer.find_dependents(changes, top)

    for filename, funcs in list(tests.items()):
        tests[filename] = [t for t in funcs if t.startswith("test_")]
        if not tests[filename]:
            del tests[filename]

    return tests

if __name__ == "__main__":
    args = diff_collator.parse_args()
    diff_output = diff_collator.get_diff_output(args)
    changes = diff_collator.parser(diff_output)
    diff_collator.output_changes(changes, args.verbosity)

    changes = {k:set(v.keys()) for k, v in  changes.items()}
    tests = select_tests(changes)
    logger.debug("tests:")
    for filename, funcs in tests.items():
        logger.debug(filename)
        for func in funcs:
            logger.debug("\t%s", func)

    #check for flakiness
    flaky = nonflaky = []
    for filename, funcs in tests.items():
        for func in funcs:
            res = flakiness_checker.run_test_trials(filename, func,
                    DEFAULT_NUM_TRIALS)
            if res != 0:
                flaky.append((filename,func))
            else:
                nonflaky.append((filename, func))

    print("Following tests failed flakiness checker:")
    if not flaky:
        print("None")
    for test in flaky:
        print("%s:%s".format(test[0], test[1]))

    print("Following tests passed flakiness checker:")
    if not nonflaky:
        print("None")
    for test in nonflaky:
        print("%s:%s".format(test[0], test[1]))

    logger.info("Tested: %d | Flaky: %d, Non-flaky: %d",
        len(flaky) + len(nonflaky), len(flaky), len(nonflaky))

