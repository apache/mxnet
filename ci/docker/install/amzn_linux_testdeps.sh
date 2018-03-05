set -ex
pip install cpplint 'pylint==1.4.4' 'astroid==1.3.6'
pip3 install nose
ln -s -f /opt/bin/nosetests /usr/local/bin/nosetests3
ln -s -f /opt/bin/nosetests-3.4 /usr/local/bin/nosetests-3.4