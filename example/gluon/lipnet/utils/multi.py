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
Module: preprocess with multi-process
"""


def multi_p_run(tot_num, _func, worker, params, n_process):
    """
    Run _func with multi-process using params.
    """
    from multiprocessing import Process, Queue
    out_q = Queue()
    procs = []

    split_num = split_seq(list(range(0, tot_num)), n_process)

    print(tot_num, ">>", split_num)

    split_len = len(split_num)
    if n_process > split_len:
        n_process = split_len

    for i in range(n_process):
        _p = Process(target=_func,
                     args=(worker, split_num[i][0], split_num[i][1],
                           params, out_q))
        _p.daemon = True
        procs.append(_p)
        _p.start()

    try:
        result = []
        for i in range(n_process):
            result.append(out_q.get())
        for i in procs:
            i.join()
    except KeyboardInterrupt:
        print('Killing all the children in the pool.')
        for i in procs:
            i.terminate()
            i.join()
        return -1

    while not out_q.empty():
        print(out_q.get(block=False))

    return result


def split_seq(sam_num, n_tile):
    """
    Split the number(sam_num) into numbers by n_tile
    """
    import math
    print(sam_num)
    print(n_tile)
    start_num = sam_num[0::int(math.ceil(len(sam_num) / (n_tile)))]
    end_num = start_num[1::]
    end_num.append(len(sam_num))
    return [[i, j] for i, j in zip(start_num, end_num)]


def put_worker(func, from_idx, to_idx, params, out_q):
    """
    put worker
    """
    succ, fail = func(from_idx, to_idx, params)
    return out_q.put({'succ': succ, 'fail': fail})


def test_worker(from_idx, to_idx, params):
    """
    the worker to test multi-process
    """
    params = params
    succ = set()
    fail = set()
    for idx in range(from_idx, to_idx):
        try:
            succ.add(idx)
        except ValueError:
            fail.add(idx)
    return (succ, fail)


if __name__ == '__main__':
    RES = multi_p_run(35, put_worker, test_worker, params={}, n_process=5)
    print(RES)
