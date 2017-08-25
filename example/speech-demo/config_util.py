import re
import os
import sys
import argparse
import mxnet as mx
import numpy as np

if sys.version_info >= (3, 0):
    import configparser
else:
    import ConfigParser as configparser


def parse_args():
    default_cfg = configparser.ConfigParser()
    default_cfg.read(os.path.join(os.path.dirname(__file__), 'default.cfg'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--configfile", help="config file for training parameters")

    # those allow us to overwrite the configs through command line
    for sec in default_cfg.sections():
        for name, _ in default_cfg.items(sec):
            arg_name = '--%s_%s' % (sec, name)
            doc = 'Overwrite %s in section [%s] of config file' % (name, sec)
            parser.add_argument(arg_name, help=doc)

    args = parser.parse_args()

    if args.configfile is not None:
        # now read the user supplied config file to overwrite some values
        default_cfg.read(args.configfile)

    # now overwrite config from command line options
    for sec in default_cfg.sections():
        for name, _ in default_cfg.items(sec):
            arg_name = ('%s_%s' % (sec, name)).replace('-', '_')
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                sys.stderr.write('!! CMDLine overwriting %s.%s:\n' % (sec, name))
                sys.stderr.write("    '%s' => '%s'\n" % (default_cfg.get(sec, name),
                                 getattr(args, arg_name)))
                default_cfg.set(sec, name, getattr(args, arg_name))

    args.config = default_cfg
    sys.stderr.write("="*80+"\n")
    return args


def get_checkpoint_path(args):
    prefix = args.config.get('train', 'prefix')
    if os.path.isabs(prefix):
        return prefix
    return os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', prefix))


def parse_contexts(args):
    # parse context into Context objects
    contexts = re.split(r'\W+', args.config.get('train', 'context'))
    for i, ctx in enumerate(contexts):
        if ctx[:3] == 'gpu':
            contexts[i] = mx.context.gpu(int(ctx[3:]))
        else:
            contexts[i] = mx.context.cpu(int(ctx[3:]))
    return contexts
