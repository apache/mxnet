#!/usr/bin/env python3

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

# -*- coding: utf-8 -*-
"""Utilities to control a guest VM, used for virtual testing with QEMU"""

__author__ = 'Pedro Larroy'
__version__ = '0.1'

import os
import sys
import subprocess
import argparse
import logging
from subprocess import call, check_call, Popen, DEVNULL, PIPE
import time
import sys
import multiprocessing
import shlex

###################################################
#
# Virtual testing with QEMU
#
# We start QEMU instances that have a local port in the host redirected to the ssh port.
#
# The VMs are provisioned after boot, tests are run and then they are stopped
#

QEMU_RUN="""
qemu-system-arm -M virt -m {ram} \
  -kernel vmlinuz \
  -initrd initrd.img \
  -append 'root=/dev/vda1' \
  -drive if=none,file=vda.qcow2,format=qcow2,id=hd \
  -device virtio-blk-device,drive=hd \
  -netdev user,id=mynet,hostfwd=tcp::{ssh_port}-:22 \
  -device virtio-net-device,netdev=mynet \
  -display none -nographic
"""

class VMError(RuntimeError):
    pass

class VM:
    """Control of the virtual machine"""
    def __init__(self, ssh_port=2222):
        self.log = logging.getLogger(VM.__name__)
        self.ssh_port = ssh_port
        self.timeout_s = 300
        self.qemu_process = None
        self._detach = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._detach:
            self.shutdown()
            self.terminate()

    def start(self):
        self.log.info("Starting VM, ssh port redirected to localhost:%s", self.ssh_port)
        if self.is_running():
            raise VMError("VM is running, shutdown first")
        self.qemu_process = run_qemu(self.ssh_port)
        def keep_waiting():
            return self.is_running()

        ssh_working = wait_ssh_open('127.0.0.1', self.ssh_port, keep_waiting, self.timeout_s)

        if not self.is_running():
            (_, stderr) = self.qemu_process.communicate()
            raise VMError("VM failed to start, retcode: {}, stderr: {}".format( self.retcode(), stderr.decode()))

        if not ssh_working:
            if self.is_running():
                self.log.error("VM running but SSH is not working")
            self.terminate()
            raise VMError("SSH is not working after {} seconds".format(self.timeout_s))
        self.log.info("VM is online and SSH is up")

    def is_running(self):
        return self.qemu_process and self.qemu_process.poll() is None

    def retcode(self):
        if self.qemu_process:
            return self.qemu_process.poll()
        else:
            raise RuntimeError('qemu process was not started')

    def terminate(self):
        if self.qemu_process:
            logging.info("send term signal")
            self.qemu_process.terminate()
            time.sleep(3)
            logging.info("send kill signal")
            self.qemu_process.kill()
            self.qemu_process.wait()
            self.qemu_process = None
        else:
            logging.warn("VM.terminate: QEMU process not running")

    def detach(self):
        self._detach = True

    def shutdown(self):
        if self.qemu_process:
            logging.info("Shutdown via ssh")
            # ssh connection will be closed with an error
            call(["ssh", "-o", "StrictHostKeyChecking=no", "-p", str(self.ssh_port), "qemu@localhost",
            "sudo", "poweroff"])
            ret = self.qemu_process.wait(timeout=90)
            self.log.info("VM on port %s has shutdown (exit code %d)", self.ssh_port, ret)
            self.qemu_process = None

    def wait(self):
        if self.qemu_process:
            self.qemu_process.wait()

    def __del__(self):
        if self.is_running and not self._detach:
            logging.info("VM destructor hit")
            self.terminate()

def run_qemu(ssh_port=2222):
    cmd = QEMU_RUN.format(ssh_port=ssh_port, ram=4096)
    logging.info("QEMU command: %s", cmd)
    qemu_process = Popen(shlex.split(cmd), stdout=DEVNULL, stdin=DEVNULL, stderr=PIPE)
    return qemu_process


def wait_ssh_open(server, port, keep_waiting=None, timeout=None):
    """ Wait for network service to appear
        @param server: host to connect to (str)
        @param port: port (int)
        @param timeout: in seconds, if None or 0 wait forever
        @return: True of False, if timeout is None may return only True or
                 throw unhandled network exception
    """
    import socket
    import errno
    import time
    log = logging.getLogger('wait_ssh_open')
    sleep_s = 0
    if timeout:
        from time import time as now
        # time module is needed to calc timeout shared between two exceptions
        end = now() + timeout

    while True:
        log.debug("Sleeping for %s second(s)", sleep_s)
        time.sleep(sleep_s)
        s = socket.socket()
        try:
            if keep_waiting and not keep_waiting():
                log.debug("keep_waiting() is set and evaluates to False")
                return False

            if timeout:
                next_timeout = end - now()
                if next_timeout < 0:
                    log.debug("connect time out")
                    return False
                else:
                    log.debug("connect timeout %d s", next_timeout)
                    s.settimeout(next_timeout)

            log.info("connect %s:%d", server, port)
            s.connect((server, port))
            ret = s.recv(1024).decode()
            if ret and ret.startswith('SSH'):
                s.close()
                log.info("wait_ssh_open: port %s:%s is open and ssh is ready", server, port)
                return True
            else:
                log.debug("Didn't get the SSH banner")
                s.close()

        except ConnectionError as err:
            log.debug("ConnectionError %s", err)
            if sleep_s == 0:
                sleep_s = 1
            else:
                sleep_s *= 2

        except socket.gaierror as err:
            log.debug("gaierror %s",err)
            return False

        except socket.timeout as err:
            # this exception occurs only if timeout is set
            if timeout:
                return False

        except TimeoutError as err:
            # catch timeout exception from underlying network library
            # this one is different from socket.timeout
            raise


def wait_port_open(server, port, timeout=None):
    """ Wait for network service to appear
        @param server: host to connect to (str)
        @param port: port (int)
        @param timeout: in seconds, if None or 0 wait forever
        @return: True of False, if timeout is None may return only True or
                 throw unhandled network exception
    """
    import socket
    import errno
    import time
    sleep_s = 0
    if timeout:
        from time import time as now
        # time module is needed to calc timeout shared between two exceptions
        end = now() + timeout

    while True:
        logging.debug("Sleeping for %s second(s)", sleep_s)
        time.sleep(sleep_s)
        s = socket.socket()
        try:
            if timeout:
                next_timeout = end - now()
                if next_timeout < 0:
                    return False
                else:
                    s.settimeout(next_timeout)

            logging.info("connect %s %d", server, port)
            s.connect((server, port))

        except ConnectionError as err:
            logging.debug("ConnectionError %s", err)
            if sleep_s == 0:
                sleep_s = 1

        except socket.gaierror as err:
            logging.debug("gaierror %s",err)
            return False

        except socket.timeout as err:
            # this exception occurs only if timeout is set
            if timeout:
                return False

        except TimeoutError as err:
            # catch timeout exception from underlying network library
            # this one is different from socket.timeout
            raise

        else:
            s.close()
            logging.info("wait_port_open: port %s:%s is open", server, port)
            return True

