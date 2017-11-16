import os
import signal
import time
import csv
import subprocess
from memory_profiler import memory_usage

IS_GPU = (os.environ['MXNET_KERAS_TEST_MACHINE'] == 'GPU')
GPU_NUM = int(os.environ['GPU_NUM']) if IS_GPU else 0

# This command is useful to fetch GPU memory consumption.
GPU_MONITOR_CMD = "nvidia-smi --query-gpu=index,memory.used --format=csv -lms 500 -f output.csv"

def cpu_memory_profile(func_to_profile):
    max_mem_usage = memory_usage(proc=(func_to_profile, ()), max_usage=True)
    return max_mem_usage[0]

def gpu_mem_profile(file_name):
    row_count = 0
    # In MBs
    max_mem_usage = 0
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        last_line_broken = False
        for row in csv_reader:
            if row_count == 0:
                row_count += 1
                continue
            if len(row) < 2 or not 'MiB' in row[1]:
                last_line_broken = True
            row_count += 1
        row_count -= 1
        if row_count % GPU_NUM == 0 and last_line_broken:
            row_count -= GPU_NUM
        else:
            row_count -= row_count % GPU_NUM

    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        current_usage = 0
        mem_recoder = [0] * GPU_NUM
        row_num = 0
        for row in csv_reader:
            if row_num == 0:
                row_num += 1
                continue
            mem_str = row[1].lstrip().rstrip()[:-4]
            mem_num = float(mem_str)
            current_usage += mem_num
            mem_recoder[(row_num - 1) % GPU_NUM] += mem_num
            if row_num % GPU_NUM == 0:
                max_mem_usage = max(max_mem_usage, current_usage)
                current_usage = 0
            row_num += 1
            if row_num > row_count:
                break
        row_num -= 1
    os.remove(file_name)
    return max_mem_usage

def profile(func_to_profile):
    """
        This function helps in profile given func_to_profile for run-time and
        memory consumption.

        Capable of profile for both GPU and CPU machine.

        Uses environment variable - IS_GPU to identify whether to profile for
        CPU or GPU.

        returns: run_time, memory_usage
    """
    run_time = 0; # Seconds
    memory_usage = 0; # MBs

    # Choose nvidia-smi or memory_profiler for memory profiling for GPU and CPU
    # machines respectively.
    if(IS_GPU):
        # Start time - For timing the runtime
        start_time = time.time()
        open('nvidia-smi-output.csv', 'a').close()
        gpu_monitor_process = subprocess.Popen(GPU_MONITOR_CMD,
                                                  shell=True, preexec_fn=os.setsid)
        func_to_profile()
        end_time = time.time()
        os.killpg(os.getpgid(gpu_monitor_process.pid), signal.SIGTERM)
        run_time = end_time - start_time
        memory_usage = gpu_mem_profile('nvidia-smi-output.csv')
    else:
        # Start time - For timing the runtime
        start_time = time.time()
        memory_usage = cpu_memory_profile(func_to_profile)
        end_time = time.time()
        run_time = end_time - start_time

    return run_time, memory_usage
