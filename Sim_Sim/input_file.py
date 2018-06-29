# input_file.py

from random import randint
# randint(1000,9999) --> when necessary
# to reproduce same data, use CONSTANT seed value
# NOTE: with multithreading to instantiate scientists, seed does not guarantee same results due to variability
# of the CPU resources at a certain time!
seed = 123456  # randint(100000, 999999)

# number of stable time_periods in the model
# NOTE: total time periods is time_periods + 2 (first two are unstable)
time_periods = 8

# Scalar: number of ideas unique to each time period
ideas_per_time = 5

# scalar: number of scientists born per time period
N = 3

# SCALAR: store means of the mean and sds for returns
true_means_lam = 300
true_sds_lam = int(0.16 * true_means_lam)

# agent constants (Use 0.03 for Windows due to GPU limit)
# possibly try 0.05*N/100?
start_effort_lam = int(0.2*true_means_lam)  # make sure this is >= 9 to ensure start_effort isn't too small

# the amount of variance from the actual perceived return
noise_factor = 1

# keep k and start effort ratio fixed (see next line)
# keep learning cost relatively low to promote new ideas (but model could still be flawed)
k_lam = int(0.3*start_effort_lam)  # make sure this is still > 0

# the number of TP a scientist can actively invest in ideas
time_periods_alive = 4

# for counting number of html pages generated / other useful counter value for debugging
count = 0

# for runtime calculations
start = 0

# whether we want parallel processing (depending if it works on the system being run)
use_multiprocessing = True

# whether to instantiate multiple scientists simultaneously
use_multithreading = False

# whether to store arrays
use_store = True

# Optimal processors by system (not exactly accurate)
# Mac w/ 4 cores --> 3
# Windows w/ 8 cores --> 3
# Windows w/ 24 cores --> 8
num_processors = 4

# whether to report all scientists in agent_df
all_scientists = False

# Potential variables for future use
start_effort_decay = int(0.2*start_effort_lam)
prop_invest_limit = 0.6  # works when <0.5?, for optimizations
