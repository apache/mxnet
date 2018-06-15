#!/bin/sh
set -e


# Install npm
apt-get install -y npm

# Install nodejs- v 7.10.1
#apt-get install -y curl python-software-properties
curl -sL https://deb.nodesource.com/setup_8.x | bash -
apt-get install -y nodejs

# Install broken link checker utility
npm install broken-link-checker -g

# Test broken link and print summary.
python test_broken_links.py
