#!/usr/bin/env python

import os, sys
import subprocess

if len(sys.argv) != 4:
  print "usage: %s <hostfile> <user> <prog>" % sys.argv[0]
  sys.exit(1)

host_file = sys.argv[1]
user = sys.argv[2]
prog_name = sys.argv[3]

kill_cmd = (
    "ps aux | "
    "grep -v grep | "
    "grep '" + prog_name + "' | "
    "awk '{if($1==\"" + user + "\")print $2;}' | "
    "xargs kill -9"
    )
print kill_cmd

# Kill program on remote machines
with open(host_file, "r") as f:
  for host in f:
    if ':' in host:
      host = host[:host.index(':')]
    print host
    subprocess.Popen(["ssh", "-oStrictHostKeyChecking=no", "%s" % host, kill_cmd],
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    print "Done killing"

# Kill program on local machine
os.system(kill_cmd)
