# input_file.py

from random import randint
# randint(1000,9999) --> when necessary
# to reproduce same data, use CONSTANT seed value
# for randomization best practice
seed = randint(100000, 999999)


# for optimizations
prop_invest_limit = 0.6  # works when <0.5?


# model variables
time_periods = 10  # stable time periods
ideas_per_time = 5
N = 6  # number of scientists alive per time period (EVEN #'s ONLY!)
max_investment_lam = 50  # based on logistic cdf with sds and means below (want lam to be near flat of top of curve (1)
true_sds_lam = 4
true_means_lam = 25


# agent constants
start_effort_lam = int(0.8*max_investment_lam)
start_effort_decay = int(0.2*start_effort_lam)

# keep k and start effort ratio fixed (see next line)
# keep learning cost relatively low to promote new ideas (but model could still be flawed)
k_lam = int(0.3*max_investment_lam)

sds_lam = true_sds_lam
means_lam = true_means_lam
time_periods_alive = 2


# life of each scientist variable?

# for counting number of html pages generated / other useful counter value for debugging
count = 0