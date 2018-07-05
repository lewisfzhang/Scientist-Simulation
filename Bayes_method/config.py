# config.py

import numpy as np
import math


# <editor-fold desc="Config Variables for Program">
# keeping seed constant ensure identical results
seed = 654321  # np.random.randint(100000, 999999)

# number of stable time_periods in the model
# NOTE: total time periods is time_periods + 2 (first two are unstable)
time_periods = 10

# Scalar: number of ideas unique to each time period
ideas_per_time = 5

# Scalar: number of scientists born per time period
# NOTE: according to jay, N should be greater than ideas_per_time
N = 3

# SCALAR: store means of the mean and sds for returns
true_means_lam = 300
true_sds_lam = int(0.4 * true_means_lam)

# agent constants (Use 0.03 for Windows due to GPU limit)
# possibly try 0.05*N/100?
start_effort_lam = int(0.5*true_means_lam)  # make sure this is >= 9 to ensure start_effort isn't too small

# keep k and start effort ratio fixed (see next line)
# keep learning cost relatively low to promote new ideas (but model could still be flawed)
k_lam = int(0.5*start_effort_lam)  # make sure this is still > 0

# the number of TP a scientist can actively invest in ideas
time_periods_alive = 6
# </editor-fold>

# <editor-fold desc="Logistics of running the program">
# whether we want parallel processing (depending if it works on the system being run)
use_multiprocessing = False

# Optimal processors by system (not exactly accurate)
# Mac w/ 4 cores --> 3
# Windows w/ 8 cores --> 3
# Windows w/ 24 cores --> 8
num_processors = 3

# handles the locations for model and agent temp objects
tmp_loc = 'tmp/'

# whether to instantiate multiple scientists simultaneously
use_multithreading = False

# whether to store arrays
use_store = False

# GUARD: user cannot do mp without using store as a protection
use_store = (use_store, True)[use_multiprocessing == True]

# whether to report all scientists in agent_df
all_scientists = False
# </editor-fold>

# <editor-fold desc="Seed manager">
# total number of scientists in model
num_scientists = N*(time_periods + 1)

# protects against noise that produces negative sds/means
end_limit = int((true_sds_lam - 3 * math.sqrt(true_sds_lam)) / 3)

# seed for constant randomization
np.random.seed(seed)
seed_array = np.random.randint(100000, 999999, (num_scientists + 1)*(time_periods + 6)).reshape(num_scientists + 1, time_periods + 6)
# </editor-fold>

# <editor-fold desc="Potential variables for future use">
start_effort_decay = int(0.2*start_effort_lam)
prop_invest_limit = 0.6  # works when <0.5?, for optimizations
# </editor-fold>

# <editor-fold desc="Global Variables for functions.py and run_graphs.py">
# for counting number of html pages generated / other useful counter value for debugging
count = 0

# for runtime calculations
start = 0

# for x,y scale
x_width = 130
y_width = 122
sq_width = 200
# </editor-fold>
