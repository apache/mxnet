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

from collections import OrderedDict

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
Spawns a thread and runs the command given in the cmd_args for specified timeout period
taken from http://stackoverflow.com/questions/1191374/using-module-subprocess-with-timeout
'''
class Command(object):
    def __init__(self, cmd_args, logfile):
        self.cmd_args = cmd_args
        self.logfile = logfile
        self.process = None

    def run(self, timeout):
        def target():
            LOGGER = logging.getLogger('benchmark')
            LOGGER.info('started running %s', ' '.join(self.cmd_args))
            log_fd = open(self.logfile, 'w')
            self.process = subprocess.Popen(self.cmd_args, stdout=log_fd, stderr=subprocess.STDOUT, universal_newlines=True)
            for line in self.process.communicate():
                LOGGER.debug(line)
            log_fd.close()
            LOGGER.info('finished running %s', ' '.join(self.cmd_args))

        LOGGER.debug('Attempting to start Thread to run %s', ' '.join(self.cmd_args))
        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            LOGGER.debug('Terminating process running %s', ' '.join(self.cmd_args))
            self.process.terminate()
            thread.join()
            time.sleep(1)

        return

log_loc = './benchmark'
LOGGER = setup_logging(log_loc)

def parse_args():
    parser = argparse.ArgumentParser(description='Run Benchmark on various imagenet networks using train_imagenent.py')
    parser.add_argument('--worker_file', type=str, help='file that contains a list of workers', required=True)
    parser.add_argument('--worker_count', type=int, help='number of workers to run benchmark on', required=True)
    parser.add_argument('--gpu_count', type=int, help='number of gpus on each worker to use', required=True)
    args = parser.parse_args()
    return args

def series(max_count):
    i=max_count
    s=[]
    while i >= 1:
        s.append(i)
        i=i/2
    return s[::-1]

class Network(object):
    def __init__(self, name, img_size, batch_size):
        self.name = name
        self.img_size = img_size
        self.batch_size = batch_size
        self.gpus_img_processed_map = {}

'''
Choose the middle iteration to get the images processed per sec
'''
def images_processed(log_loc):
    f=open(log_loc)
    img_per_sec = re.findall("(?:Batch\s+\[30\]\\\\tSpeed:\s+)(\d+\.\d+)(?:\s+)", str(f.readlines()))
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

def stop_old_processes(hosts_file):
    stop_args = ['python', '../../tools/kill-mxnet.py', hosts_file]
    stop_args_str = ' '.join(stop_args)
    LOGGER.info('killing old remote processes\n %s', stop_args_str)
    stop = subprocess.check_output(stop_args, stderr=subprocess.STDOUT)
    LOGGER.debug(stop)
    time.sleep(1)

def run_imagenet(kv_store, data_shape, batch_size, num_gpus, num_nodes, network, args_workers_file):
    imagenet_args=['python',  'train_imagenet.py',  '--gpus', ','.join(str(i) for i in xrange(num_gpus)), \
                   '--network', network, '--batch-size', str(batch_size * num_gpus), \
                   '--data-shape', str(data_shape), '--num-epochs', '1' ,'--kv-store', kv_store, '--benchmark']
    log = log_loc + '/' + network + '_' + str(num_nodes*num_gpus)
    hosts = log + '_workers'
    generate_hosts_file(num_nodes, hosts, args_workers_file)
    stop_old_processes(hosts)
    launch_args = ['../../tools/launch.py', '-n', str(num_nodes), '-s', str(num_nodes*2), '-H', hosts, ' '.join(imagenet_args) ]

    #user train_imagenet when running on a single node
    if kv_store == 'device':
        imagenet = Command(imagenet_args, log)
        imagenet.run(600)
    else:
        launch = Command(launch_args, log)
        launch.run(600)

    stop_old_processes(hosts)
    img_per_sec = images_processed(log)
    LOGGER.info('network: %s, num_gpus: %d, image/sec: %f', network, num_gpus*num_nodes, img_per_sec)
    return img_per_sec

def main():
    args = parse_args()
    speedup_chart = pygal.Line(x_title ='gpus',y_title ='images/sec', logarithmic=True)
    speedup_chart.x_labels = map(str, series(args.worker_count* args.gpu_count))
    speedup_chart.add('ideal speedup', series(128))
    networks= [
        Network('inception-v3',img_size=299, batch_size=32)
        ,Network('resnet', img_size=224, batch_size=2)
        ,Network('alexnet', img_size=224, batch_size=1024)
    ]

    for net in networks:
        # to run on all the gpus, we will 'dist_sync_device' as the kv_store option,
        # hence dropping in this loop from the generated series
        for num_gpus in series(args.gpu_count)[:-1]:
            imgs_per_sec = run_imagenet(kv_store='device', data_shape=net.img_size, batch_size=net.batch_size, \
                                        num_gpus=num_gpus, num_nodes=1, network=net.name, args_workers_file=args.worker_file)
            net.gpus_img_processed_map[num_gpus] = imgs_per_sec
        for num_nodes in series(args.worker_count):
            imgs_per_sec = run_imagenet(kv_store='dist_sync_device', data_shape=net.img_size, batch_size=net.batch_size, \
                         num_gpus=args.gpu_count, num_nodes=num_nodes, network=net.name, args_workers_file=args.worker_file)
            net.gpus_img_processed_map[num_nodes * args.gpu_count] = imgs_per_sec

        d = OrderedDict(sorted(net.gpus_img_processed_map.items(), key=lambda t: t[0]))
        img_processed_on_single_gpu = d.values()[0]
        speedup_chart.add(net.name , [ each/img_processed_on_single_gpu for each in d.values()], formatter=lambda x: 'speedup:%d, img/sec:%.2f, batch/gpu:%d' % (x, x*img_processed_on_single_gpu, net.batch_size))
        LOGGER.info('Network: %s (num_gpus, images_processed): %s', net.name, ','.join(map(str, d.items())))
        with open(log_loc + '/' + net.name + '.csv', 'wb') as f:
            w = csv.writer(f)
            w.writerow(['num_gpus', 'img_processed_per_sec'])
            w.writerows(d.items())

    speedup_chart.render_to_file(log_loc + '/speedup.svg')

if __name__ == '__main__':
    main()
