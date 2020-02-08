#!/usr/bin/env python

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
Launch a distributed job for BytePS
Combining the byteps/launcher/dist_launcher.py and byteps/launcher/launch.py of 
https://github.com/bytedance/byteps.git @ 2152d88
"""


import argparse
import os, sys
import signal
import logging
import subprocess
from multiprocessing import Pool, Process
from threading import Thread

def preprocess_envs(args_envs):
    envs_map = {}
    for item in args_envs:
        i = item.find(":")
        if i != -1:
            key = item[:i]
            val = item[i+1:]
        envs_map[key] = val
    return envs_map

def get_env(envs_map):
    envs = []
    # get system envs
    keys = ['OMP_NUM_THREADS', 'KMP_AFFINITY']
    for k in keys:
        v = os.getenv(k)
        if v is not None:
            envs.append('export ' + k + '=' + v + ';')
    # get ass_envs
    for k, v in envs_map.items():
        envs.append('export ' + str(k) + '=' + str(v) + ';')
    return (' '.join(envs))

def get_hosts_from_file(filename):
    with open(filename) as f:
        tmp = f.readlines()
    assert len(tmp) > 0
    hosts = []
    for h in tmp:
        if len(h.strip()) > 0:
            # parse addresses of the form ip:port
            h = h.strip()
            i = h.find(":")
            p = "22"
            if i != -1:
                p = h[i+1:]
                h = h[:i]
            # hosts now contain the pair ip, port
            hosts.append((h, p))
    return hosts

def start_ssh(prog, node, port, username, fname):
    def run(prog):
        subprocess.check_call(prog, shell=True)

    dirname = 'sshlog'
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    pname = dirname + '/' + fname
    if username is not None:
        prog = 'ssh -o StrictHostKeyChecking=no ' + ' -l ' + username \
               + ' ' + node + ' -p ' + port + ' \'' + prog + '\'' \
               + ' > ' + pname + '.stdout' + ' 2>' + pname + '.stderr&'
    else:
        prog = 'ssh -o StrictHostKeyChecking=no ' + node + ' -p ' + port + ' \'' + prog + '\'' \
               + ' > ' + pname + '.stdout' + ' 2>' + pname + '.stderr&'

    thread = Thread(target=run, args=(prog,))
    thread.setDaemon(True)
    thread.start()
    return thread

def submit(args):
    if args.server_hostfile is not None:
        server_hosts = get_hosts_from_file(args.server_hostfile)
        worker_hosts = get_hosts_from_file(args.hostfile)
        args.num_workers = len(worker_hosts)
        args.num_servers = len(server_hosts)
    else:
        assert (args.num_servers is not None and args.num_workers is not None), \
                "For BytePS backend, you must specify num_servers and num_workers"
        all_hosts = get_hosts_from_file(args.hostfile)
        assert(len(all_hosts) == args.num_workers + args.num_servers ), \
        "The sum of the number of workers and servers must be equal to \
        the number of hosts in the hostfile"
        server_hosts = all_hosts[:args.num_servers]
        worker_hosts = all_hosts[args.num_servers:]
    
    num_server = args.num_servers
    num_worker = args.num_workers
    assert num_server >= 1, "There must be at least one server."
    assert num_worker >= 1, "There must be at least one worker."

    print('Launch %d workers and %d servers' % (num_worker, num_server))

    # common env
    pass_envs = preprocess_envs(args.env)
    pass_envs['DMLC_NUM_WORKER'] = str(num_worker)
    pass_envs['DMLC_NUM_SERVER'] = str(num_server)
    pass_envs['DMLC_PS_ROOT_URI'] = str(args.scheduler_ip) # This ip is localhost
    pass_envs['DMLC_PS_ROOT_PORT'] = str(args.scheduler_port) # This port is allocated automatically.
    
    username = None
    threads = []
    for (node, port) in [(args.scheduler_ip, str(22))]:
        name = 'scheduler'
        pass_envs['DMLC_ROLE'] = name
        print('Laucnhing Scheduler...')
        prog = get_env(pass_envs) + (" python3 -c " + "\"" + "import byteps.server" + "\"")
        threads.append(start_ssh(prog, node, port, username, name))
    
    for i, (node, port) in enumerate(worker_hosts):
        name = 'worker'
        pass_envs['DMLC_ROLE'] = name
        pass_envs['DMLC_WORKER_ID'] = str(i)
        print('Laucnhing Worker{} ...'.format(i))
        if "NVIDIA_VISIBLE_DEVICES" in os.environ:
            local_size = len(os.environ["NVIDIA_VISIBLE_DEVICES"].split(","))
        else:
            local_size = 1
        for local_rank in range(local_size):
            pass_envs["BYTEPS_LOCAL_RANK"] = str(local_rank)
            pass_envs["BYTEPS_LOCAL_SIZE"] = str(local_size)
            command = args.command
            if int(os.getenv("BYTEPS_ENABLE_GDB", 0)):
                if command.find("python3") != 0:
                    command = "python3 " + command
                command = ["gdb -ex 'run' -ex 'bt' -batch --args "] + command
            prog = get_env(pass_envs) + (' '.join(command))

            if pass_envs["BYTEPS_TRACE_ON"] == "1":
                print("\n!!!Enable profiling for WORKER_ID: %s and local_rank: %d!!!" % (pass_envs["DMLC_WORKER_ID"], local_rank))
                print("BYTEPS_TRACE_START_STEP: %s\tBYTEPS_TRACE_END_STEP: %s\t BYTEPS_TRACE_DIR: %s" % (pass_envs["BYTEPS_TRACE_START_STEP"], os.environ.get("BYTEPS_TRACE_END_STEP", ""), os.environ.get("BYTEPS_TRACE_DIR", "")))
                print("Command: %s\n" % command)
                sys.stdout.flush()
                trace_path = os.path.join(pass_envs["BYTEPS_TRACE_DIR"], str(local_rank))
                if not os.path.exists(trace_path):
                    os.makedirs(trace_path)
            threads.append(start_ssh(prog, node, port, username, name + str(i)))
    
    for i, (node, port) in enumerate(server_hosts):
        name = 'server'
        pass_envs['DMLC_ROLE'] = name
        print('Laucnhing Server{} ...'.format(i))
        prog = get_env(pass_envs) + (" python3 -c " + "\"" + "import byteps.server" + "\"")
        threads.append(start_ssh(prog, node, port, username, name + str(i)))

    for t in threads:
        t.join()