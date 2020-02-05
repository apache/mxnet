#!/usr/bin/python

from __future__ import print_function
import os
import subprocess
import threading
import sys
import time

COMMON_REQUIRED_ENVS = ["DMLC_ROLE", "DMLC_NUM_WORKER", "DMLC_NUM_SERVER",
                        "DMLC_PS_ROOT_URI", "DMLC_PS_ROOT_PORT"]
WORKER_REQUIRED_ENVS = ["DMLC_WORKER_ID"]

def check_env():
    assert "DMLC_ROLE" in os.environ and \
           os.environ["DMLC_ROLE"].lower() in ["worker", "server", "scheduler"]
    required_envs = COMMON_REQUIRED_ENVS
    if os.environ["DMLC_ROLE"] == "worker":
        assert "DMLC_NUM_WORKER" in os.environ
        num_worker = int(os.environ["DMLC_NUM_WORKER"])
        assert num_worker >= 1
        if num_worker == 1:
            required_envs = []
        required_envs += WORKER_REQUIRED_ENVS
    for env in required_envs:
        if env not in os.environ:
            print("The env " + env + " is missing")
            os._exit(0)

def worker(local_rank, local_size, command):
    my_env = os.environ.copy()
    my_env["BYTEPS_LOCAL_RANK"] = str(local_rank)
    my_env["BYTEPS_LOCAL_SIZE"] = str(local_size)
    if int(os.getenv("BYTEPS_ENABLE_GDB", 0)):
        if command.find("python") != 0:
            command = "python " + command
        command = "gdb -ex 'run' -ex 'bt' -batch --args " + command

    if os.environ.get("BYTEPS_TRACE_ON", "") == "1":
        print("\n!!!Enable profiling for WORKER_ID: %s and local_rank: %d!!!" % (os.environ.get("DMLC_WORKER_ID"), local_rank))
        print("BYTEPS_TRACE_START_STEP: %s\tBYTEPS_TRACE_END_STEP: %s\t BYTEPS_TRACE_DIR: %s" % (os.environ.get("BYTEPS_TRACE_START_STEP", ""), os.environ.get("BYTEPS_TRACE_END_STEP", ""), os.environ.get("BYTEPS_TRACE_DIR", "")))
        print("Command: %s\n" % command)
        sys.stdout.flush()
        trace_path = os.path.join(os.environ.get("BYTEPS_TRACE_DIR", "."), str(local_rank))
        if not os.path.exists(trace_path):
            os.makedirs(trace_path)
    subprocess.check_call(command, env=my_env, stdout=sys.stdout, stderr=sys.stderr, shell=True)

if __name__ == "__main__":
    print("BytePS launching " + os.environ["DMLC_ROLE"])
    sys.stdout.flush()
    check_env()
    if os.environ["DMLC_ROLE"] == "worker":
        if "NVIDIA_VISIBLE_DEVICES" in os.environ:
            local_size = len(os.environ["NVIDIA_VISIBLE_DEVICES"].split(","))
        else:
            local_size = 1
        t = [None] * local_size
        for i in range(local_size):
            command = ' '.join(sys.argv[1:])
            t[i] = threading.Thread(target=worker, args=[i, local_size, command])
            t[i].daemon = True
            t[i].start()

        for i in range(local_size):
            t[i].join()

    else:
        import byteps.server