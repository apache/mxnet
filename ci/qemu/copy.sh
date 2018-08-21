#!/bin/bash
set -ex
virt-copy-out -a vda.qcow2 /boot/vmlinuz-3.16.0-6-armmp-lpae /boot/initrd.img-3.16.0-6-armmp-lpae .
