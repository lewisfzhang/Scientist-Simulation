#!/bin/bash
# run simulation and data collection
# if succeed, open result web page

# check params (probably using python if statements to check this is more ideal
# [ $# -gt 0 -a $# -lt 7 ] && echo $0 need 7 parameter values | tee -a ../data/output.txt && exit 1

curdir=$(dirname $0)
cd $curdir

# create new output.txt file
echo > ../data/output.txt

parentdir=$HOME  # $HOME is user home directory
venvdir=$(cd $parentdir && pwd)/venv
venvact=$venvdir/bin/activate
[ ! -e $venvact ] && [ -x ./install.sh ] && ./install.sh $venvdir 
[ -d src/tmp ] && rm -r src/tmp
(source $venvact; python3 run.py $* && python3 collect.py)
if [ $? -eq 0 ]; then
	# open ../data/pages/all_images.html
	# open ../data/pages/page_ideas.html
    # open ../data/pages/page_agent_vars.html
	echo *** Succeed *** | tee -a ../data/output.txt
	echo | tee -a ../data/output.txt

	cp -r tmp/model ../data/tmp_model
    if [ $# -eq 0 ]; then
        echo tar -C ../data -czf ../data/zipped/run_$(echo 'default_params').tar.gz images pages parameters.txt output.txt saved tmp_model | tee -a ../data/output.txt
        /usr/bin/tar -C ../data -czf ../data/zipped/run_$(echo 'default_params').tar.gz images pages parameters.txt output.txt saved tmp_model
    else
        echo tar -C ../data -czf ../data/zipped/run_$(echo $* | tr ' ' '_').tar.gz images pages parameters.txt output.txt saved tmp_model | tee -a ../data/output.txt
        /usr/bin/tar -C ../data -czf ../data/zipped/run_$(echo $* | tr ' ' '_').tar.gz images pages parameters.txt output.txt saved tmp_model
    fi

    echo "removing agent related objects"
    rm -r ../data/tmp_model
    [ -e tmp/agent_1 ] && rm -r tmp/agent_*
else
	echo !!! Error !!! | tee -a ../data/output.txt
fi
