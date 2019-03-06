<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# QEMU base image creation

This folder contains scripts and configuration to create a QEMU virtual drive with a debian system.

The order of execution is:
- `init.sh` to download the installation kernel and ramdisk
- `preseed.sh` to preseed the debian installer so it doesn't ask questions 
- `copy.sh` to extract the kernel and ramdisk from the installed system
- `run.sh` to boot the system and fine tune the image

# Description of the process:

# Preparing the base image

First, an installation is made using installer kernel and initrd by using the scripts above.

# After installation, we extract initrd and kernel from the installation drive

The commands look like this:

`virt-copy-out -a hda.qcow2 /boot/initrd.img-4.15.0-30-generic-lpae .`

In the same way for the kernel.

Then we install packages and dependencies on the qemu image:

apt install -y sudo python3-dev virtualenv wget libgfortran3 libopenblas-base rsync build-essential
libopenblas-dev libomp5

We enable sudo and passwordless logins:

Add file `/etc/sudoers.d/01-qemu`
With content:
```
qemu ALL=(ALL) NOPASSWD: ALL
```

Edit: `/etc/ssh/sshd_config`

And set the following options:
```
PermitEmptyPasswords yes
PasswordAuthentication yes
PermitRootLogin yes
```

Disable root and user passwords with `passwd -d`

Edit ` /etc/pam.d/common-auth`

Replace `auth    [success=1 default=ignore]      pam_unix.so nullok_secure` by 
```
auth    [success=1 default=ignore]      pam_unix.so nullok
```

As root to install system wide:

```
wget -nv https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
apt-get clean
```

Afterwards install mxnet python3 deps:

```
pip3 install -r mxnet_requirements.txt
```


To access qemu control console from tmux: `ctrl-a a c`
