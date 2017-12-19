import subprocess, os
import argparse
import logging

parser = argparse.ArgumentParser(description="Run sparse linear classification " \
                                             "with distributed kvstore",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu', type=int, default=0,
                    help='which gpu')


num_gpus = 8

args = parser.parse_args()
gpu = args.gpu
print("total num gpus = %d, current gpu = %d" % (num_gpus, gpu))

LR = [1]
LR_DECAY = [0.25, 0.5]
BPTT = [35]
K = [60]
CLIP = [0.2]
DROPOUT = [0.5]
MOM = [0.9, 0.95]
BETA1 = [0.9]
WD = [0, 1e-5]
HID = 650
USE_DENSE = False
OPTIM = 'sgd'

niter = 0
total_iter = len(LR) * len(LR_DECAY) * len(BPTT) * len(K) * len(CLIP) * len(DROPOUT) * len(MOM) * len(WD)

for lr in LR:
    for bptt in BPTT:
        for k in K:
            for clip in CLIP:
                for dropout in DROPOUT:
                    for lr_decay in LR_DECAY:
                        for mom in MOM:
                            for wd in WD:
                                for beta1 in BETA1:
                                    if niter % num_gpus == gpu:
                                        my_env = os.environ.copy()
                                        my_env["PYTHONPATH"] = "/home/ubuntu/nce/python:" + my_env["PYTHONPATH"]

                                        config = ["--bptt", str(bptt), "--k", str(k), '--dropout', str(dropout), \
                                                  '--clip', str(clip), '--lr', str(lr), '--lr-decay', str(lr_decay), \
                                                  '--mom', str(mom), '--wd', str(wd), '--optimizer', str(OPTIM), '--beta1', str(beta1)]
                                        if USE_DENSE:
                                            config += ['--use-dense']
                                        cmd = ["python", "train.py", "--nhid", str(HID), "--emsize", str(HID), "--log-interval=999999", \
                                               "--num-gpus=1", "--gpu=%d" % gpu, '--epoch=80', '--tied'] + config
                                        filename = 'logs-sparse/tune' + '-'.join(config) + ".tunelog"
                                        with open(filename, "w") as outfile:
                                            subprocess.check_call(cmd, stderr=outfile, env=my_env)
                                            #subprocess.call(cmd)
                                    niter += 1
                                    print("%d of %d" %(niter, total_iter))
