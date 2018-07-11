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
[ -d src/tmp ] && rm -r src/tmp
(source $venvact; python3 run.py $* && python3 collect.py)
if [ $? -eq 0 ]; then
	open ../data/pages/all_images.html
	open ../data/pages/page_ideas.html
    open ../data/pages/page_agent_vars.html
	echo *** Succeed ***
	echo
    if [ $# -eq 0 ]; then
        echo tar -C ../data -czf ../data/zipped/run_$(echo 'default_params').tar.gz images pages parameters.txt
        /usr/bin/tar -C ../data -czf ../data/zipped/run_$(echo 'default_params').tar.gz images pages parameters.txt
    else
        echo tar -C ../data -czf ../data/zipped/run_$(echo $* | tr ' ' '_').tar.gz images pages parameters.txt
        /usr/bin/tar -C ../data -czf ../data/zipped/run_$(echo $* | tr ' ' '_').tar.gz images pages parameters.txt
    fi
else
	echo !!! Error !!!
fi
