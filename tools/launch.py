#!/usr/bin/env python
"""
Launch a distributed job
"""
import argparse
import os, sys
import signal
import logging

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../dmlc-core/tracker"))

def dmlc_opts(opts):
    """convert from mxnet's opts to dmlc's opts
    """
    args = ['--num-workers', str(opts.num_workers),
            '--num-servers', str(opts.num_servers),
            '--cluster', opts.launcher,
            '--host-file', opts.hostfile,
            '--sync-dst-dir', opts.sync_dst_dir]
    args += opts.command;
    try:
        from dmlc_tracker import opts
    except ImportError:
        print("Can't load dmlc_tracker package.  Perhaps you need to run")
        print("    git submodule update --init --recursive")
        raise
    dmlc_opts = opts.get_opts(args)
    return dmlc_opts


def main():
    parser = argparse.ArgumentParser(description='Launch a distributed job')
    parser.add_argument('-n', '--num-workers', required=True, type=int,
                        help = 'number of worker nodes to be launched')
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
    parser.add_argument('--launcher', type=str, default='ssh',
                        choices = ['local', 'ssh', 'mpi', 'sge', 'yarn'],
                        help = 'the launcher to use')
    parser.add_argument('command', nargs='+',
                        help = 'command for launching the program')
    args, unknown = parser.parse_known_args()
    args.command += unknown
    if args.num_servers is None:
        args.num_servers = args.num_workers

    args = dmlc_opts(args)
    
    if args.host_file is None or args.host_file == 'None':
      if args.cluster == 'yarn':
          from dmlc_tracker import yarn
          yarn.submit(args)
      elif args.cluster == 'local':
          from dmlc_tracker import local
          local.submit(args)
      elif args.cluster == 'sge':
          from dmlc_tracker import sge
          sge.submit(args)
      else:
          raise RuntimeError('Unknown submission cluster type %s' % args.cluster)
    else:
      if args.cluster == 'ssh':
          from dmlc_tracker import ssh
          ssh.submit(args)
      elif args.cluster == 'mpi':
          from dmlc_tracker import mpi
          mpi.submit(args)
      else:
          raise RuntimeError('Unknown submission cluster type %s' % args.cluster)

def signal_handler(signal, frame):
    logging.info('Stop launcher')
    sys.exit(0)

if __name__ == '__main__':
    fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    signal.signal(signal.SIGINT, signal_handler)
    main()
