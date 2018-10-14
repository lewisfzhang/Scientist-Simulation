#!/bin/bash
# run master.py while ensuring python modules are properly installed before doing so

curdir=$(dirname $0)
cd $curdir

# create new output.txt file
echo > ../data/output.txt

parentdir=$HOME  # $HOME is user home directory
venvdir=$(cd $parentdir && pwd)/venv
venvact=$venvdir/bin/activate
[ ! -e $venvact ] && [ -x ./install.sh ] && ./install.sh $venvdir
[ -d src/tmp ] && rm -r src/tmp
(source $venvact; python3 master.py)
echo "DONE!"
