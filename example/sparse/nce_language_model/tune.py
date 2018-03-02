import subprocess, os
import argparse
import logging

parser = argparse.ArgumentParser(description="Run sparse linear classification " \
                                             "with distributed kvstore",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu', type=int, default=0,
                    help='which gpu')


num_gpus = 1

args = parser.parse_args()
gpu = args.gpu
print("total num gpus = %d, current gpu = %d" % (num_gpus, gpu))

LR = [0.05, 0.1, 0.2]
BPTT = [20, 35]
K = [8192, 10240]
CLIP = [1, 5, 10, 15]
DROPOUT = [0.01, 0.05, 0.1, 0.2]

#LR = [0.2]
#BPTT = [20]
#K = [8192]
#CLIP = [10]
#DROPOUT = [0.1]
total_iter = len(LR) * len(BPTT) * len(K) * len(CLIP) * len(DROPOUT)

for lr in LR:
    for bptt in BPTT:
        for k in K:
            for clip in CLIP:
                for dropout in DROPOUT:
                    my_env = os.environ.copy()
                    my_env["PYTHONPATH"] = "/home/ubuntu/tf/python:" + my_env["PYTHONPATH"]

                    config = ["--bptt", str(bptt), "--k", str(k), '--dropout', str(dropout), \
                              '--clip', str(clip), '--lr', str(lr)]
                    cmd = ["python", "train.py", "--gpus=0,1,2,3", "--data=/home/ubuntu/gbw-5/training-monolingual.tokenized.shuffled/*",
                           "--per-ctx-clip", "--epoch=1", "--checkpoint-interval=100"] + config
                    filename = 'logs-5/tune' + '-'.join(config) + ".tunelog"
                    with open(filename, "w") as outfile:
                        subprocess.check_call(cmd, stderr=outfile, env=my_env)
                    print("%d of %d" %(niter, total_iter))
