#!/bin/bash
#set -xeuo pipefail
set -ex
#cp installer-initrd.gz initrd
#gunzip installer-initrd.gz
rm -rf initrd
mkdir -p initrd
cd initrd
gunzip -c ../installer-initrd.gz | cpio -i 
cp ../preseed.cfg .
cp ../initrd_modif/inittab etc/inittab
cp ../initrd_modif/S10syslog lib/debian-installer-startup.d/S10syslog
find .  | cpio --create --format 'newc'  | gzip -c > ../installer-initrd_automated.gz
#find .  | cpio --create --format 'newc'  | gzip -c > ../installer-initrd.gz
#echo preseed.cfg | cpio -H newc -o -A -F installer-initrd 
#gzip installer-initrd
#cd ..
#cp initrd/installer-initrd.gz .
echo "Done!"
