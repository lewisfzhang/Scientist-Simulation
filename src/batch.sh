#!/bin/bash

#SBATCH --job-name=scientist_simulation       # Job name
#SBATCH --mail-type=ALL                       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=lewis.zhang19@bcp.org     # Where to send mail
#SBATCH --ntasks=1                            # Run on a single CPU
#SBATCH --mem=10gb                            # Job memory request
#SBATCH --time=00:05:00                       # Time limit hrs:min:sec
#SBATCH --output=~/batch/batch_%j.log         # Standard output and error log

pwd; hostname; date

ml reset
ml load python/3.6.1

echo "Running run.sh script"

storage_dir = $PI_HOME/run/$(date +%d-%b-%H_%M)
mkdir $storage_dir

time_periods = 50
ideas_per_time = 20
N = 40
time_periods_alive = 10
true_means_lam = 300

for prop_sds in {0.2..0.8..0.2}
do
    for prop_means in {0.25..0.75..0.25}
    do
        for prop_start in 0.1 0.25 0.4 0.5 0.6 0.75
        do
            cp -r /home/users/lewisz/Scientist-Simulation/ $storage_dir/run_$(echo $time_periods $ideas_per_time $N $time_periods_alive $true_means_lam $prop_sds $prop_means $prop_start)
            srun $storage_dir/src/run.sh $time_periods $ideas_per_time $N $time_periods_alive $true_means_lam $prop_sds $prop_means $prop_start
        done
    done
done

date
