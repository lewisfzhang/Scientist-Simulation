#!/bin/bash

#SBATCH --job-name=run_batch_($cat $HOME/batch/job_count.txt)
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=4
#SBATCH --mem=10gb
#SBATCH --time=00:20:00
#SBATCH --output=$HOME/batch/run_$(cat $HOME/batch/init.txt | tr ' ' '_').log

pwd; hostname; date

ml reset
ml load python/3.6.1

cd $storage_dir/run_$(cat $HOME/batch/init.txt | tr ' ' '_')/src

./run.sh $(cat tmp/init.txt)

date

exit 0
