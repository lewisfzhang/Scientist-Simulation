#!/bin/bash

#SBATCH --job-name=batch.sh                   # Job name
#SBATCH --mail-type=END,FAIL                  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=lewis.zhang19@bcp.org     # Where to send mail
#SBATCH --ntasks=1                            # Run on a single CPU
#SBATCH --mem=1gb                             # Job memory request
#SBATCH --time=00:05:00                       # Time limit hrs:min:sec
#SBATCH --output=~/batch/batch_%j.log         # Standard output and error log

pwd; hostname; date

ml reset
ml load python/3.6.1

echo "Running run.sh script"

storage_dir=$PI_HOME/lewisz/run/$(SLURM_JOBID)_$(date +%d-%b-%H_%M)
mkdir $storage_dir

time_periods=50
ideas_per_time=20
N=40
time_periods_alive=10
count=1

for prop_sds in {0.2..0.8..0.2}
do
    for prop_means in {0.25..0.75..0.25}
    do
        for prop_start in 0.1 0.25 0.4 0.5 0.6 0.75
        do
            echo $time_periods $ideas_per_time $N $time_periods_alive $prop_sds $prop_means $prop_start > $HOME/batch/init.txt
            echo $count > $HOME/batch/job_count.txt
            count=$((count + 1))
            cp -r $HOME/Scientist-Simulation/ $storage_dir/run_$(cat $HOME/batch/init.txt | tr ' ' '_')
            sbatch $storage_dir/src/run_batch.sh
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
