# input_file.py

from random import randint
# randint(1000,9999) --> when necessary
# to reproduce same data, use CONSTANT seed value
# for randomization best practice
seed = randint(100000, 999999)
num_processors = 5

# model variables
time_periods = 10  # stable time periods
ideas_per_time = 1
N = 20  # number of scientists alive per time period (EVEN #'s ONLY!)
true_means_lam = 300
true_sds_lam = int(0.16 * true_means_lam)

# agent constants
start_effort_lam = int((0.5*N/100)*true_means_lam)  # make sure this is >= 9 to ensure start_effort isn't too small
noise_factor = 1

# keep k and start effort ratio fixed (see next line)
# keep learning cost relatively low to promote new ideas (but model could still be flawed)
k_lam = int(0.4*start_effort_lam)  # make sure this is still > 0

sds_lam = true_sds_lam
means_lam = true_means_lam
time_periods_alive = 4


# for counting number of html pages generated / other useful counter value for debugging
count = 0

# for runtime calculations
start = 0


# VARIABLES WE DON'T WANT ANYMORE
max_investment_lam = 50  # based on logistic cdf with sds and means below (want lam to be near flat of top of curve (1)
start_effort_decay = int(0.2*start_effort_lam)
prop_invest_limit = 0.6  # works when <0.5?, for optimizations
