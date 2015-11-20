#!/usr/bin/env python
"""
Launch a distributed job
"""
import argparse
import os, sys
import signal
import logging

def main():
    parser = argparse.ArgumentParser(description='Launch a distributed job')
    parser.add_argument('-n', '--num-workers', required=True, type=int,
                        help = 'number of worker nodes to be launched')
    parser.add_argument('-s', '--num-servers', type=int,
                        help = 'number of server nodes to be launched, \
                        in default it is equal to NUM_WORKERS')
    parser.add_argument('-H', '--hostfile', type=str,
                        help = 'the hostfile of slave machines which will run the job')
    parser.add_argument('--sync-dir', type=str,
                        help = 'if specificed, it will sync the current \
                        directory into slave machines\'s SYNC_DIR')
    parser.add_argument('--launcher', type=str, default='ssh',
                        choices = ['ssh', 'mpirun'],
                        help = 'the lancher to use')
    parser.add_argument('command', nargs='+',
                        help = 'command for launching the program')
    args, unknown = parser.parse_known_args()

    if args.num_servers is None:
        args.num_servers = args.num_workers

    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(curr_path, "../ps-lite/tracker"))

    if args.hostfile is None:
        from dmlc_local import LocalLauncher
        launcher = LocalLauncher(args, unknown)
    elif args.launcher == 'ssh':
        from dmlc_ssh import SSHLauncher
        launcher = SSHLauncher(args, unknown)
    else:
        return

    launcher.run()

def signal_handler(signal, frame):
    logging.info('Stop luancher')
    sys.exit(0)

if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal_handler)

    main()
