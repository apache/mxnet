#!/usr/bin/env python

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
import os, sys
import subprocess

if len(sys.argv) != 4:
  print("usage: %s <hostfile> <user> <prog>" % sys.argv[0])
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
print(kill_cmd)

# Kill program on remote machines
with open(host_file, "r") as f:
  for host in f:
    if ':' in host:
      host = host[:host.index(':')]
    print(host)
    subprocess.Popen(["ssh", "-oStrictHostKeyChecking=no", "%s" % host, kill_cmd],
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    print("Done killing")

# Kill program on local machine
os.system(kill_cmd)
