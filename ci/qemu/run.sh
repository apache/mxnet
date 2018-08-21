#!/bin/bash
set -ex
disk=${1:-vda.qcow2}
qemu-system-arm -M virt -m 1024 \
  -kernel vmlinuz-3.16.0-6-armmp-lpae \
  -initrd initrd.img-3.16.0-6-armmp-lpae \
  -append 'root=/dev/vda1' \
  -drive if=none,file=$disk,format=qcow2,id=hd \
  -device virtio-blk-device,drive=hd \
  -netdev user,id=mynet \
  -device virtio-net-device,netdev=mynet \
  -nographic
