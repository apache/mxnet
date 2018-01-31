from __future__ import absolute_import
import yaml
import collections
import os.path as osp
import logging

CONFIG = {}
DEFAULT_CONFIG = osp.join(osp.dirname(__file__), 'default.yml')


def get_config():
    """Grab the config as dict()."""
    return CONFIG

def load_config(cfg_file):
    """Update configurations with new config file."""
    cfg = get_config()
    with open(cfg_file, 'r') as inf:
        new_cfg = yaml.load(inf)
    if new_cfg:
        cfg.update(new_cfg)
    else:
        logging.warning('Nothing loaded from %s', cfg_file)
    return cfg

def update_config(new_cfg):
    """Update configs with dict."""
    def recursive_update(d, u, log=''):
        for k, v in u.items():
            if isinstance(d, collections.Mapping):
                if not k in d:
                    logging.warning('%s is not in default config, is it on purpose?',
                                    log + '.' + str(k))
                if isinstance(v, collections.Mapping):
                    r = recursive_update(d.get(k, {}), v, log=log + '.' + str(k))
                    d[k] = r
                else:
                    d[k] = u[k]
            else:
                logging.warning('%s is not a parent', str(d))
        return d
    cfg = get_config()
    recursive_update(cfg, new_cfg)
    return cfg

def save_config(filename):
    """Save current configuration to file."""
    with open(filename, 'w') as fout:
        fout.write(yaml.dump(get_config(), default_flow_style=False))

def dump_config():
    return yaml.dump(get_config(), default_flow_style=False)

# load the default configurations
CONFIG = load_config(DEFAULT_CONFIG)
