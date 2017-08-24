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

from __future__ import print_function
import logging
import argparse
import os
import time
import sys
import shutil
import csv
import re
import subprocess, threading
import pygal
import importlib
import collections
import threading
import copy
'''
Setup Logger and LogLevel
'''
def setup_logging(log_loc):
    if os.path.exists(log_loc):
        shutil.move(log_loc, log_loc + "_" + str(int(os.path.getctime(log_loc))))
    os.makedirs(log_loc)

    log_file = '{}/benchmark.log'.format(log_loc)
    LOGGER = logging.getLogger('benchmark')
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s %(message)s')
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)
    return LOGGER

'''
Runs the command given in the cmd_args for specified timeout period
and terminates after
'''
class RunCmd(threading.Thread):
    def __init__(self, cmd_args, logfile):
        threading.Thread.__init__(self)
        self.cmd_args = cmd_args
        self.logfile = logfile
        self.process = None

    def run(self):
        LOGGER = logging.getLogger('benchmark')
        LOGGER.info('started running %s', ' '.join(self.cmd_args))
        log_fd = open(self.logfile, 'w')
        self.process = subprocess.Popen(self.cmd_args, stdout=log_fd, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in self.process.communicate():
            LOGGER.debug(line)
        log_fd.close()
        LOGGER.info('finished running %s', ' '.join(self.cmd_args))

    def startCmd(self, timeout):
        LOGGER.debug('Attempting to start Thread to run %s', ' '.join(self.cmd_args))
        self.start()
        self.join(timeout)
        if self.is_alive():
            LOGGER.debug('Terminating process running %s', ' '.join(self.cmd_args))
            self.process.terminate()
            self.join()
            time.sleep(1)
        return

log_loc = './benchmark'
LOGGER = setup_logging(log_loc)

class Network(object):
    def __init__(self, mode, name, img_size, batch_size):
        self.mode = mode
        self.name = name
        self.img_size = img_size
        self.batch_size = batch_size
        self.gpu_speedup = collections.OrderedDict()

def parse_args():
    class NetworkArgumentAction(argparse.Action):
        def validate(self, attrs):
            args = attrs.split(':')
            if len(args) != 4 or isinstance(args[0], str) == False or isinstance(args[1], str) == False:
                print('expected network attributes in format mode:network_name:batch_size:image_size \
                \nThe network_name is a valid model defined as network_name.py in the image-classification/symbol folder. \
                \nOr a gluon vision model defined in mxnet/python/mxnet/gluon/model_zoo/model_store.py.')
                sys.exit(1)
            try:
                # check if the network exists
                if args[0] == 'native':
                    importlib.import_module('symbols.' + args[1])
                batch_size = int(args[2])
                img_size = int(args[3])
                return Network(mode=args[0], name=args[1], batch_size=batch_size, img_size=img_size)
            except Exception as e:
                print('expected network attributes in format mode:network_name:batch_size:image_size \
                \nThe network_name is a valid model defined as network_name.py in the image-classification/symbol folder. \
                \nOr a gluon vision model defined in mxnet/python/mxnet/gluon/model_zoo/model_store.py.')
                print(e)
                sys.exit(1)

        def __init__(self, *args, **kw):
            kw['nargs'] = '+'
            argparse.Action.__init__(self, *args, **kw)

        def __call__(self, parser, namespace, values, option_string=None):
            if isinstance(values, list) == True:
                setattr(namespace, self.dest, map(self.validate, values))
            else:
                setattr(namespace, self.dest, self.validate(values))

    parser = argparse.ArgumentParser(description='Run Benchmark on various imagenet networks using train_imagenent.py')
    parser.add_argument('--networks', dest='networks', nargs='+', type=str, help='one or more networks in the format mode:network_name:batch_size:image_size \
    \nThe network_name is a valid model defined as network_name.py in the image-classification/symbol folder for native imagenet \
    \n Or a gluon vision model defined in mxnet/python/mxnet/gluon/model_zoo/model_store.py.',
                        action=NetworkArgumentAction)
    parser.add_argument('--worker_file', type=str,
                        help='file that contains a list of worker hostnames or list of worker ip addresses that can be sshed without a password.',
                        required=True)
    parser.add_argument('--worker_count', type=int, help='number of workers to run benchmark on.', required=True)
    parser.add_argument('--gpu_count', type=int, help='number of gpus on each worker to use.', required=True)
    args = parser.parse_args()
    return args

def series(max_count):
    i = 1
    s = []
    while i <= max_count:
        s.append(i)
        i = i * 2
    if s[-1] < max_count:
        s.append(max_count)
    return s

'''
Choose the middle iteration to get the images processed per sec
'''
def images_processed(log_loc, mode):
    f = open(log_loc)
    if mode == 'native':
        img_per_sec = re.findall("(?:Batch\s+\[30\]\\\\tSpeed:\s+)(\d+\.\d+)(?:\s+)", str(f.readlines()))
    else:
        img_per_sec = re.findall("(?:Batch\s+\[3\]\\\\tSpeed:\s+)(\d+\.\d+)(?:\s+)", str(f.readlines()))
    f.close()
    img_per_sec = map(float, img_per_sec)
    total_img_per_sec = sum(img_per_sec)
    return total_img_per_sec

def generate_hosts_file(num_nodes, workers_file, args_workers_file):
    f = open(workers_file, 'w')
    output = subprocess.check_output(['head', '-n', str(num_nodes), args_workers_file])
    f.write(output)
    f.close()
    return

def stop_old_processes(hosts_file, prog_name):
    stop_args = ['python', '../../tools/kill-mxnet.py', hosts_file, 'python', prog_name]
    stop_args_str = ' '.join(stop_args)
    LOGGER.info('killing old remote processes\n %s', stop_args_str)
    stop = subprocess.check_output(stop_args, stderr=subprocess.STDOUT)
    LOGGER.debug(stop)
    time.sleep(1)

def run_benchmark(kv_store, data_shape, batch_size, num_gpus, num_nodes, network, args_workers_file, mode):
    if mode == 'native':
        benchmark_args = ['python', 'train_imagenet.py', '--gpus', ','.join(str(i) for i in range(num_gpus)), \
                          '--network', network, '--batch-size', str(batch_size * num_gpus), \
                          '--image-shape', '3,' + str(data_shape) + ',' + str(data_shape), '--num-epochs', '1',
                          '--kv-store', kv_store, '--benchmark', '1', '--disp-batches', '10']
    else:
        benchmark_args = ['python', '../gluon/image_classification.py', '--dataset', 'dummy', '--gpus', str(num_gpus), \
                          '--epochs', '1', '--benchmark', '--mode', mode, '--model', network, '--batch-size',
                          str(batch_size), \
                          '--log-interval', str(1), '--kvstore', kv_store]

    log = log_loc + '/' + network + '_' + str(num_nodes * num_gpus) + '_log'
    hosts = log_loc + '/' + network + '_' + str(num_nodes * num_gpus) + '_workers'
    generate_hosts_file(num_nodes, hosts, args_workers_file)
    if mode == 'native':
        stop_old_processes(hosts, 'train_imagenet.py')
    else:
        stop_old_processes(hosts, '../gluon/image-classification.py')
    launch_args = ['../../tools/launch.py', '-n', str(num_nodes), '-s', str(num_nodes * 2), '-H', hosts,
                   ' '.join(benchmark_args)]

    # use train_imagenet/image_classification when running on a single node
    if kv_store == 'device':
        imagenet = RunCmd(benchmark_args, log)
        imagenet.startCmd(timeout=60 * 10)
    else:
        launch = RunCmd(launch_args, log)
        launch.startCmd(timeout=60 * 10)

    if mode == 'native':
        stop_old_processes(hosts, 'train_imagenet.py')
    else:
        stop_old_processes(hosts, '../gluon/image-classification.py')
    img_per_sec = images_processed(log, mode)
    LOGGER.info('network: %s, num_gpus: %d, image/sec: %f', network, num_gpus * num_nodes, img_per_sec)
    return img_per_sec

def plot_graph(args):
    speedup_chart = pygal.Line(x_title='gpus', y_title='speedup', logarithmic=True)
    speedup_chart.x_labels = map(str, series(args.worker_count * args.gpu_count))
    speedup_chart.add('ideal speedup', series(args.worker_count * args.gpu_count))
    for net in args.networks:
        image_single_gpu = net.gpu_speedup[1] if 1 in net.gpu_speedup or not net.gpu_speedup[1] else 1
        y_values = [each / image_single_gpu for each in net.gpu_speedup.values()]
        LOGGER.info('%s: image_single_gpu:%.2f' % (net.name, image_single_gpu))
        LOGGER.debug('network:%s, y_values: %s' % (net.name, ' '.join(map(str, y_values))))
        speedup_chart.add(net.name, y_values \
            , formatter=lambda y_val, img=copy.deepcopy(image_single_gpu), batch_size=copy.deepcopy(
            net.batch_size): 'speedup:%.2f, img/sec:%.2f, batch/gpu:%d' % \
            (0 if y_val is None else y_val, 0 if y_val is None else y_val * img, batch_size))
    speedup_chart.render_to_file(log_loc + '/speedup.svg')

def write_csv(log_loc, args):
    for net in args.networks:
        with open(log_loc + '/' + net.name + '.csv', 'wb') as f:
            w = csv.writer(f)
            w.writerow(['num_gpus', 'img_processed_per_sec'])
            w.writerows(net.gpu_speedup.items())

def main():
    args = parse_args()
    for net in args.networks:
        # use kv_store='device' when running on 1 node
        for num_gpus in series(args.gpu_count):
            imgs_per_sec = run_benchmark(kv_store='device', data_shape=net.img_size, batch_size=net.batch_size, \
                                         num_gpus=num_gpus, num_nodes=1, network=net.name,
                                         args_workers_file=args.worker_file, mode=net.mode)
            net.gpu_speedup[num_gpus] = imgs_per_sec
        for num_nodes in series(args.worker_count)[1::]:
            imgs_per_sec = run_benchmark(kv_store='dist_sync_device', data_shape=net.img_size,
                                         batch_size=net.batch_size, \
                                         num_gpus=args.gpu_count, num_nodes=num_nodes, network=net.name,
                                         args_workers_file=args.worker_file, mode=net.mode)
            net.gpu_speedup[num_nodes * args.gpu_count] = imgs_per_sec
        LOGGER.info('Network: %s (num_gpus, images_processed): %s', net.name, ','.join(map(str, net.gpu_speedup.items())))
    write_csv(log_loc, args)
    plot_graph(args)

if __name__ == '__main__':
    main()
