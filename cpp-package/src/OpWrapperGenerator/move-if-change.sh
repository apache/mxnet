#!/bin/sh
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <source file> <destination file>"
fi

if [ ! -f "$2" ]; then
    mv -v "$1" "$2"
    exit 0
fi

diff "$1" "$2" >/dev/null

if [ $? -ne 0 ]; then
    mv -v "$1" "$2"
else
    rm -f "$1"
fi

