#!/usr/bin/env python

import os, sys

if len(sys.argv) != 2:
  print "usage: %s <hostfile>" % sys.argv[0]
  sys.exit(1)

host_file = sys.argv[1]
prog_name = "train_imagenet"

# Get host IPs
with open(host_file, "r") as f:
  hosts = f.read().splitlines()
ssh_cmd = (
    "ssh "
    "-o StrictHostKeyChecking=no "
    "-o UserKnownHostsFile=/dev/null "
    "-o LogLevel=quiet "
    )
kill_cmd = (
    " "
    "ps aux |"
    "grep -v grep |"
    "grep 'python train_imagenet.py' |"
    "awk '{print \$2}'|"
    "xargs kill"
    )
print kill_cmd
for host in hosts:
  cmd = ssh_cmd + host +" \""+ kill_cmd+"\""
  print cmd
  os.system(cmd)

  print "Done killing"
