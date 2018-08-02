import logging
import signal
import os
import subprocess
import json

import flakiness_checker
import diff_collator
import dependency_analyzer

LOGGING_FILE = os.path.join(os.path.dirname(__file__), "results.log")

TESTS_DIRECTORY = "tests/python"
TEST_TRIALS = 10000
TIME_BUDGET = 60 #7200 # 2 hours, in seconds

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

fh = logging.FileHandler(LOGGING_FILE)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
logger.addHandler(fh)
 
def select_tests(changes):
    """returns tests that are dependent on given changes
    """

    top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    top = top.decode("utf-8").splitlines()[0]

    tests = dependency_analyzer.find_dependents(changes, top)

    for filename, funcs in list(tests.items()):
        tests[filename] = [t for t in funcs if t.startswith("test_")]
        if not tests[filename]:
            del tests[filename]

    return tests

def check_tests(tests):
    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, handler)
    
    flaky, nonflaky = [], []
    q = TIME_BUDGET // len(tests)
    for t in tests:
        try:
            signal.alarm(q)
            res = flakiness_checker.run_test_trials(t[0], t[1], TEST_TRIALS)
            signal.alarm(0)
        except TimeoutError:
            res = 0
            logger.warning("flakiness checker exceeded time budget")

        if res != 0:
            flaky.append(t)
        else:
            nonflaky.append(t)
    
    return flaky, nonflaky



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = diff_collator.parse_args()
    diff_output = diff_collator.get_diff_output(args)
    changes = diff_collator.parser(diff_output)

    changes = {k:set(v.keys()) for k, v in  changes.items()}
    tests = select_tests(changes)
    logger.debug("tests:")
    for filename, funcs in tests.items():
        logger.debug(filename)
        for func in funcs:
            logger.debug("\t%s", func)

    #check for flakiness
    flaky, nonflaky = [], []
    if tests:
        flaky, nonflaky = check_tests(
            [(fi, fu) for fi in tests.keys() for fu in tests[fi]])

    print("Following tests failed flakiness checker:")
    if not flaky:
        print("None")
    for test in flaky:
        print("%s:%s".format(test[0], test[1]))

    print("Following tests passed flakiness checker:")
    if not nonflaky:
        print("None")
    for test in nonflaky:
        print("{}:{}".format(test[0], test[1]))

    logger.info("Tested: %d | Flaky: %d, Non-flaky: %d",
        len(flaky) + len(nonflaky), len(flaky), len(nonflaky))

    if flaky:
        sys.exit(1)
