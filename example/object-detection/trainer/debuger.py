import numpy as np

def super_print(*args):
    threshold = np.get_printoptions().get('threshold', 1000)
    np.set_printoptions(threshold=np.inf)
    for arg in args:
        print(arg)
    np.set_printoptions(threshold=threshold)

def find_abnormal(arr):
    pos = np.where(not np.isfinite(arr))
    return pos
