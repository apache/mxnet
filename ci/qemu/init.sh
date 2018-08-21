#!/bin/bash
set -ex
wget -O installer-vmlinuz http://http.us.debian.org/debian/dists/jessie/main/installer-armhf/current/images/netboot/vmlinuz
wget -O installer-initrd.gz http://http.us.debian.org/debian/dists/jessie/main/installer-armhf/current/images/netboot/initrd.gz
