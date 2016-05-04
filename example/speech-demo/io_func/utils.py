import sys, subprocess, pickle, os, json, logging, socket
import logging.config
import datetime

from . import info

def getRunDir():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def setup_logger(logging_ini):
    if logging_ini is not None:
        print("Using custom logger")
    else:
        logging_ini = os.path.join(info.CONFIGS, 'logging.ini')

    logging.config.fileConfig(logging_ini)
    logger = logging.getLogger(__name__)
    logger.info("**************************************************")
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    logger.info("Host:   " + str(socket.gethostname()))
    logger.info("Screen: " + os.getenv("STY", "unknown"))
    logger.info("PWD:    " + os.getenv("PWD", "unknown"))
    logger.info("Cmd:    " + str(sys.argv))
    logger.info("**************************************************")

def to_bool(obj):
    if str(obj).lower() in ["true", "1"]:
        return True
    elif str(obj).lower() in ["false", "0"]:
        return False
    else:
        raise Exception("to_bool: cannot convert to bool")

def line_with_arg(line):
    line = line.strip()
    return line is not "" and not line.startswith("#")

def parse_conv_spec(conv_spec, batch_size):
    # "1x29x29:100,5x5,p2x2:200,4x4,p2x2,f"
    conv_spec = conv_spec.replace('X', 'x')
    structure = conv_spec.split(':')
    conv_layer_configs = []
    for i in range(1, len(structure)):
        config = {}
        elements = structure[i].split(',')
        if i == 1:
            input_dims = structure[i - 1].split('x')
            prev_map_number = int(input_dims[0])
            prev_feat_dim_x = int(input_dims[1])
            prev_feat_dim_y = int(input_dims[2])
        else:
            prev_map_number = conv_layer_configs[-1]['output_shape'][1]
            prev_feat_dim_x = conv_layer_configs[-1]['output_shape'][2]
            prev_feat_dim_y = conv_layer_configs[-1]['output_shape'][3]

        current_map_number = int(elements[0])
        filter_xy = elements[1].split('x')
        filter_size_x = int(filter_xy[0])
        filter_size_y = int(filter_xy[1])
        pool_xy = elements[2].replace('p','').replace('P','').split('x')
        pool_size_x = int(pool_xy[0])
        pool_size_y = int(pool_xy[1])
        output_dim_x = (prev_feat_dim_x - filter_size_x + 1) / pool_size_x
        output_dim_y = (prev_feat_dim_y - filter_size_y + 1) / pool_size_y

        config['input_shape'] = (batch_size, prev_map_number, prev_feat_dim_x, prev_feat_dim_y)
        config['filter_shape'] = (current_map_number, prev_map_number, filter_size_x, filter_size_y)
        config['poolsize'] = (pool_size_x, pool_size_y)
        config['output_shape'] = (batch_size, current_map_number, output_dim_x, output_dim_y)
        if len(elements) == 4 and elements[3] == 'f':
            config['flatten'] = True
        else:
            config['flatten'] = False

        conv_layer_configs.append(config)
    return conv_layer_configs

def _relu(x):
    return x * (x > 0)

def _capped_relu(x):
    return T.minimum(x * (x > 0), 6)

def _linear(x):
    return x * 1.0

def parse_activation(act_str):
    print("***", act_str)
    if act_str == 'sigmoid':
        return T.nnet.sigmoid
    elif act_str == 'tanh':
        return T.tanh
    elif act_str == 'relu':
        return _relu
    elif act_str == 'capped_relu':
        return _capped_relu
    elif act_str == 'linear':
        return _linear
    return T.nnet.sigmoid

def activation_to_txt(act_func):
    if act_func == T.nnet.sigmoid:
        return 'sigmoid'
    if act_func == T.tanh:
        return 'tanh'

def parse_two_integers(argument_str):
    elements = argument_str.split(":")
    int_strs = elements[1].split(",")
    return int(int_strs[0]), int(int_strs[1])

"""
Usage:
    command = 'mysqladmin create test -uroot -pmysqladmin12'
    for line in run_command(command):
        print(line)
"""
def run_command(command):
    fnull = open(os.devnull, 'w')
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=fnull,
                         shell=True)
    return p, iter(p.stdout.readline, b'')

def pickle_load(filename):
    f = open(filename, "rb")
    try:
        obj = pickle.load(f)
    except Exception:
        f.close()
        f = open(filename, "rb")
        print("Not a pickled file... try to load as text format: " + filename)
        obj = json.load(f)
    f.close()
    return obj

def pickle_save(obj, filename):
    f = open(filename + ".new", "wb")
    pickle.dump(obj, f)
    f.close()
    os.rename(filename + ".new", filename)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def kahan_add(total, carry, inc):
    cs = T.add_no_assoc(carry, inc)
    s = T.add_no_assoc(total, cs)
    update_carry = T.sub(cs, T.sub(s, total))
    update_total = s
    return update_total, update_carry
