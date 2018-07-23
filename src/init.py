# init.py
# default config is 50 20 40 10 0.4 0.5 0.25

import numpy as np

# keeping seed constant ensure identical results
seed = 311016  # np.random.randint(100000, 999999)

# number of stable time_periods in the model
# NOTE: total time periods is time_periods + 2 (first two are unstable)
time_periods = 10

# Scalar: number of ideas unique to each time period
ideas_per_time = 5

# Scalar: number of scientists born per time period
# NOTE: according to jay, N should be greater than ideas_per_time
N = 5

# the number of TP a scientist can actively invest in ideas
time_periods_alive = 4

# SCALAR: store means of the mean and sds for returns
true_means_lam = 300

# SCALAR: proportion of sds to means
prop_sds = 0.4

# SCALAR: proportion of start effort to means of idea
prop_means = 0.5

# SCALAR: proportion of k to start effort
prop_start = 0.25

# temp location of this specific run (for batch runs)
tmp_loc = 'tmp/'
