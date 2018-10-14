#!/bin/bash
[ $# -eq 0 ] && echo "Usage: $0 </path/to/new/virtual/environment>"  | tee -a ../data/output.txt && exit 0
pkgs="IPython Mesa matplotlib numpy scipy flask requests tensorflow sklearn"
echo set up virtual environment in $(cd $(dirname $1) && pwd)/venv | tee -a ../data/output.txt
echo with packages: $pkgs | tee -a ../data/output.txt
unset PYTHONPATH
([ -f /etc/lsb-release ] && sudo apt-get install git python3-venv python3-tk
python3 -m venv $1 && $1/bin/pip install --upgrade pip && $1/bin/pip install $pkgs) > install.log
#echo PATH=$PATH | tee -a ../data/output.txt
