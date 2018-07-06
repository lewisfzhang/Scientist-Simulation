#!/bin/bash
# run simulation and data collection
# if succeed, open result web page

curdir=$(dirname $0)
cd $curdir
venvdir=../venv
bindir=$venvdir/bin
pkgdir=$venvdir/lib/python3.7/site-packages
[ ! -d $venvdir ] && echo missing venv directory : $venvdir && exit 1
[ ! -d $bindir ] && echo missing bin directory : $bindir && exit 2
[ ! -d $pkgdir ] && echo missing packages directory : $pkgdir && exit 3
[ -d tmp ] && rm -r tmp
export PATH=$PATH:$bindir
export PYTHONPATH=$pkgdir
python3 run.py && python3 collect.py
if [ $? -eq 0 ]; then
	open $curdir/web/pages/all_images.html
	echo *** Succeed ***
	echo tar czf pages.##_##_##_##.tar.gz images pages parameters.txt
else
	echo !!! Error !!!
fi
