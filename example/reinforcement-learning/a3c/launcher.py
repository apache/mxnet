"""Submission job for local jobs."""
# pylint: disable=invalid-name
from __future__ import absolute_import

import sys
import os
import subprocess
import logging
from threading import Thread
import argparse
import signal

sys.path.append(os.path.join(os.environ['HOME'], "mxnet/dmlc-core/tracker"))
sys.path.append(os.path.join('/scratch', "mxnet/dmlc-core/tracker"))
from dmlc_tracker import tracker

keepalive = """
nrep=0
rc=254
while [ $rc -ne 0 ];
do
    export DMLC_NUM_ATTEMPT=$nrep
    %s
    rc=$?;
    nrep=$((nrep+1));
done
"""

def exec_cmd(cmd, role, taskid, pass_env):
    """Execute the command line command."""
    if cmd[0].find('/') == -1 and os.path.exists(cmd[0]) and os.name != 'nt':
        cmd[0] = './' + cmd[0]
    cmd = ' '.join(cmd)
    env = os.environ.copy()
    for k, v in pass_env.items():
        env[k] = str(v)

    env['DMLC_TASK_ID'] = str(taskid)
    env['DMLC_ROLE'] = role
    env['DMLC_JOB_CLUSTER'] = 'local'

    ntrial = 0
    while True:
        if os.name == 'nt':
            env['DMLC_NUM_ATTEMPT'] = str(ntrial)
            ret = subprocess.call(cmd, shell=True, env=env)
            if ret != 0:
                ntrial += 1
                continue
        else:
            bash = cmd
            ret = subprocess.call(bash, shell=True, executable='bash', env=env)
        if ret == 0:
            logging.debug('Thread %d exit with 0', taskid)
            return
        else:
            if os.name == 'nt':
                sys.exit(-1)
            else:
                raise RuntimeError('Get nonzero return code=%d' % ret)

def submit(args):
    gpus = args.gpus.strip().split(',')
    """Submit function of local jobs."""
    def mthread_submit(nworker, nserver, envs):
        """
        customized submit script, that submit nslave jobs, each must contain args as parameter
        note this can be a lambda function containing additional parameters in input

        Parameters
        ----------
        nworker: number of slave process to start up
        nserver: number of server nodes to start up
        envs: enviroment variables to be added to the starting programs
        """
        procs = {}
        for i, gpu in enumerate(gpus):
            for j in range(args.num_threads):
                procs[i] = Thread(target=exec_cmd, args=(args.command + ['--gpus=%s'%gpu], 'worker', i*args.num_threads+j, envs))
                procs[i].setDaemon(True)
                procs[i].start()
        for i in range(len(gpus)*args.num_threads, len(gpus)*args.num_threads + nserver):
            procs[i] = Thread(target=exec_cmd, args=(args.command, 'server', i, envs))
            procs[i].setDaemon(True)
            procs[i].start()

    # call submit, with nslave, the commands to run each job and submit function
    tracker.submit(args.num_threads*len(gpus), args.num_servers, fun_submit=mthread_submit,
                   pscmd=(' '.join(args.command)))

def signal_handler(signal, frame):
    logging.info('Stop launcher')
    sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch a distributed job')
    parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('-n', '--num-threads', required=True, type=int,
                        help = 'number of threads per gpu')
    parser.add_argument('-s', '--num-servers', type=int,
                        help = 'number of server nodes to be launched, \
                        in default it is equal to NUM_WORKERS')
    parser.add_argument('-H', '--hostfile', type=str,
                        help = 'the hostfile of slave machines which will run \
                        the job. Required for ssh and mpi launcher')
    parser.add_argument('--sync-dst-dir', type=str,
                        help = 'if specificed, it will sync the current \
                        directory into slave machines\'s SYNC_DST_DIR if ssh \
                        launcher is used')
    parser.add_argument('--launcher', type=str, default='local',
                        choices = ['local', 'ssh', 'mpi', 'sge', 'yarn'],
                        help = 'the launcher to use')
    parser.add_argument('command', nargs='+',
                        help = 'command for launching the program')
    args, unknown = parser.parse_known_args()
    args.command += unknown
    if args.num_servers is None:
        args.num_servers = args.num_threads * len(args.gpus.strip().split(','))

    signal.signal(signal.SIGINT, signal_handler)
    submit(args)
