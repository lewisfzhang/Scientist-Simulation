#!/bin/bash
# run simulation and data collection
# if succeed, open result web page

# check params
[ $# -gt 0 -a $# -lt 4 ] && echo $0 need 4 parameter values && exit 1

curdir=$(dirname $0)
cd $curdir
venvdir=$(cd .. && pwd)/venv
venvact=$venvdir/bin/activate
[ ! -e $venvact ] && [ -x ./install.sh ] && ./install.sh $venvdir 
[ -d tmp ] && rm -r tmp
(source $venvact; python3 run.py $* && python3 collect.py)
if [ $? -eq 0 ]; then
	open $curdir/web/pages/all_images.html
	echo *** Succeed ***
	echo
	echo tar -C web -czf web/pages.$(echo $* | tr ' ' '_').tar.gz images pages parameters.txt
	/usr/bin/tar -C web -czf web/pages.$(echo $* | tr ' ' '_').tar.gz images pages parameters.txt
	#/usr/bin/tar -C web -cf - images pages parameters.txt | gzip > pages.$(echo $* | tr ' ' '_').tar.gz
else
	echo !!! Error !!!
fi
