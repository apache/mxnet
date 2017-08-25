import sys, os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '../common/'))
sys.path.insert(0, os.path.join(curr_path, '../../../python'))

import models
import get_data


def assertRaises(expected_exception, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except expected_exception as e:
        pass
    else:
        # Did not raise exception
        assert False, "%s did not raise %s" % (func.__name__, expected_exception.__name__)
