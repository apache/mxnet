#!/bin/bash
set -ex
rm -f vda.qcow2
sudo ./preseed.sh
qemu-img create -f qcow2 vda.qcow2 10G
qemu-system-arm -M virt -m 1024 \
  -kernel installer-vmlinuz \
  -append BOOT_DEBUG=2,DEBIAN_FRONTEND=noninteractive \
  -initrd installer-initrd_automated.gz \
  -drive if=none,file=vda.qcow2,format=qcow2,id=hd \
  -device virtio-blk-device,drive=hd \
  -netdev user,id=mynet \
  -device virtio-net-device,netdev=mynet \
  -nographic -no-reboot
