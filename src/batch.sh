#!/bin/bash

#SBATCH --job-name=batch.sh                   # Job name
#SBATCH --mail-type=END,FAIL                  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=lewis.zhang19@bcp.org     # Where to send mail
#SBATCH --ntasks=1                            # Run on a single CPU
#SBATCH --mem=100mb                           # Job memory request
#SBATCH --time=00:10:00                       # Time limit hrs:min:sec
#SBATCH --output=debug/batch_%j.log     # Standard output and error log

pwd; hostname; date

ml reset
ml load python/3.6.1

echo "Running batch.sh script"

storage_dir=$PI_HOME/lewisz/batch_$(date +%d-%b-%H_%M)
echo $storage_dir > $HOME/batch/storage_dir.txt
storage_dir=$(cat $HOME/batch/storage_dir.txt)
mkdir $storage_dir

time_periods=50
ideas_per_time=20
N=40
time_periods_alive=10
count=1

for prop_sds in 0.2 0.4 0.6 0.8
do
    for prop_means in 0.25 0.5 0.75
    do
        for prop_start in 0.1 0.25 0.4 0.5 0.6 0.75
        do
            echo $time_periods $ideas_per_time $N $time_periods_alive $prop_sds $prop_means $prop_start > $HOME/batch/init.txt
            echo $count > $HOME/batch/job_count.txt
            count=$((count + 1))
            echo $d to $storage_dir/run_$(cat $HOME/batch/init.txt | tr ' ' '_')
            for d in $PI_HOME/lewisz/storage_batch/*/; do mv $d $storage_dir/run_$(cat $HOME/batch/init.txt | tr ' ' '_'); break; done
            # cp -r $HOME/Scientist-Simulation/ $storage_dir/run_$(cat $HOME/batch/init.txt | tr ' ' '_')
            echo $(cat $HOME/batch/init.txt) > $storage_dir/run_$(cat $HOME/batch/init.txt | tr ' ' '_')/src/tmp/init.txt
            sh $storage_dir/run_$(cat $HOME/batch/init.txt | tr ' ' '_')/src/run_batch.sh
        done
    done
done


# squeue -u lewisz > queue.txt
# num=wc -l queue.txt
# while [ $num -ne 1 ]; do
#     sleep 5  # sleep 5 seconds
#     squeue -u lewisz > queue.txt
#     num=wc -l queue.txt
# done
#
# rm queue.txt
echo "all tasks completed"
date
