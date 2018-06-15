#!/bin/sh
set -e


# Install npmi
echo "Installing npm"
sudo apt-get install -y npm

# Install nodejs- v 7.10.1
#apt-get install -y curl python-software-properties
echo "Obtaining NodeJS version 8.x"
curl -sL https://deb.nodesource.com/setup_8.x | bash -

echo "Installing nodejs"
sudo apt-get install -y nodejs

# Install broken link checker utility
npm install broken-link-checker -g

# Test broken link and print summary.
echo "Running test_broken_links.py"
python test_broken_links.py
