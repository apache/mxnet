#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

"""Multi arch dockerized build tool.

"""

__author__ = 'Pedro Larroy'
__version__ = '0.1'

import os
import sys
import subprocess
import logging
import argparse
from subprocess import check_call
import glob
import re

class CmdResult(object):
    def __init__(self, std_out, std_err, status_code):
        self.std_out = std_out
        self.std_err = std_err
        self.status_code = status_code if status_code is not None else 0

    def __str__(self):
        return "%s, %s, %s" % (self.std_out, self.std_err, self.status_code)

def run(cmd, fail_on_error=True):
    logging.debug("executing shell command:\n" + cmd)
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    std_out, std_err = proc.communicate()
    if fail_on_error:
        if proc.returncode != 0:
            logging.warn('Error running command: {}'.format(cmd))
        assert proc.returncode == 0, std_err
    res = CmdResult(std_out.decode('utf-8'), std_err.decode('utf-8'), proc.returncode)
    return res


def mkdir_p(d):
    rev_path_list = list()
    head = d
    while len(head) and head != os.sep:
        rev_path_list.append(head)
        (head, tail) = os.path.split(head)

    rev_path_list.reverse()
    for p in rev_path_list:
        try:
            os.mkdir(p)
        except OSError as e:
            if e.errno != 17:
                raise

def get_arches():
    """Get a list of architectures given our dockerfiles"""
    dockerfiles = glob.glob("Dockerfile.build.*")
    dockerfiles = list(filter(lambda x: x[-1] != '~', dockerfiles))
    arches = list(map(lambda x: re.sub(r"Dockerfile.build.(.*)", r"\1", x), dockerfiles))
    arches.sort()
    return arches

def sync_source():
    logging.info("Copying sources")
    check_call(["rsync","-a","--delete","--exclude=\".git/\"",'--exclude=/docker_multiarch/',"../","mxnet"])

def get_docker_tag(arch):
    return "mxnet.build.{0}".format(arch)

def get_dockerfile(arch):
    return "Dockerfile.build.{0}".format(arch)

def build(arch):
    """Build the given architecture in the container"""
    assert arch in get_arches(), "No such architecture {0}, Dockerfile.build.{0} not found".format(arch)
    logging.info("Building for target platform {0}".format(arch))
    check_call(["docker", "build",
        "-f", get_dockerfile(arch),
        "-t", get_docker_tag(arch),
        "."])

def collect_artifacts(arch):
    """Collects the artifacts built inside the docker container to the local fs"""
    def artifact_path(arch):
        return "{}/build/{}".format(os.getcwd(), arch)
    logging.info("Collect artifacts from build in {0}".format(artifact_path(arch)))
    mkdir_p("build/{}".format(arch))

    # Mount artifact_path on /$arch inside the container and copy the build output so we can access
    # locally from the host fs
    check_call(["docker","run",
        "-v", "{}:/{}".format(artifact_path(arch), arch),
        get_docker_tag(arch),
        "bash", "-c", "cp -r /work/build/* /{}".format(arch)])

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)-15s %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch",
        help="Architecture",
        type=str)

    parser.add_argument("-l", "--list_arch",
        help="List architectures",
        action='store_true')
    args = parser.parse_args()

    if args.list_arch:
        arches = get_arches()
        print(arches)

    elif args.arch:
        sync_source()
        build(args.arch)
        collect_artifacts(args.arch)

    else:
        arches = get_arches()
        logging.info("Building for all architectures: {}".format(arches))
        logging.info("Artifacts will be produced in the build/ directory.")
        sync_source()
        for arch in arches:
            build(arch)
            collect_artifacts(arch)

    return 0

if __name__ == '__main__':
    sys.exit(main())

