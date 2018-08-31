# QEMU base image creation

This folder contains scripts and configuration to create a QEMU virtual drive with a debian system.

The order of execution is:
- `init.sh` to download the installation kernel and ramdisk
- `preseed.sh` to preseed the debian installer so it doesn't ask questions 
- `copy.sh` to extract the kernel and ramdisk from the installed system
- `run.sh` to boot the system and fine tune the image
