import numpy as np

def super_print(*args):
    threshold = np.get_printoptions().get('threshold', 1000)
    np.set_printoptions(threshold=np.inf)
    for arg in args:
        print(arg)
    np.set_printoptions(threshold=threshold)

def find_abnormal(arr):
    pos = np.where(np.logical_not(np.isfinite(arr)))
    if pos[0].size < 1:
        return None
    return pos
