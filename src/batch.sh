#!/bin/bash

#SBATCH --job-name=batch.sh                   # Job name
#SBATCH --mail-type=END,FAIL                  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=lewis.zhang19@bcp.org     # Where to send mail
#SBATCH --ntasks=4                            # Run on a single CPU
#SBATCH --mem=10gb                            # Job memory request
#SBATCH --time=00:30:00                       # Time limit hrs:min:sec
#SBATCH --output=debug/batch_%j.log           # Standard output and error log

# if we are running on supercomputer
if [ $(cd ~ && pwd) == /home/users/lewisz ]; then
    ml reset
    ml load python/3.6.1
fi

echo "running batch.sh script"

cd $(git rev-parse --show-toplevel)/src

if [ $# -gt 0 -a $# -lt 4 ]; then
    echo $0 need 4 parameter values
    exit 1
fi

if [ $# -eq 4 ]; then
    time_periods=$1
    ideas_per_time=$2
    N=$3
    time_periods_alive=$4
elif [ $# -eq 0 ]; then
    time_periods=50
    ideas_per_time=20
    N=40
    time_periods_alive=10
fi

count=1

for prop_sds in 0.2 0.4 0.6 0.8
do
    for prop_means in 0.25 0.5 0.75
    do
        for prop_start in 0.1 0.25 0.4 0.5 0.6 0.75
        do
            echo $time_periods $ideas_per_time $N $time_periods_alive $prop_sds $prop_means $prop_start > tmp/init.txt
            echo -e "\n\n\n"trial $count with configurations $(cat tmp/init.txt)"\n\n"
            ./run.sh $(cat tmp/init.txt)
            count=$((count+1))
        done
    done
done

echo "finished running batch.sh script"
