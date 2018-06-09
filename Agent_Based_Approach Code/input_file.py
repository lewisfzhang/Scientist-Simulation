# input_file.py

from random import randint
# randint(1000,9999) --> when necessary
# to reproduce same data, use CONSTANT seed value
# for randomization best practice
seed = randint(1000,9999)

# for optimizations
prop_invest_limit = 0.6  # works when <0.5?

# model variables
time_periods = 5
ideas_per_time = 1  # haven't implemented >1...
N = 2  # number of scientists per time period, for now just stick to 2
max_investment_lam = 30
true_sds_lam = 4
true_means_lam = 25

# agent constants
start_effort_lam = 15
start_effort_decay = 2
k_lam = 10  # keep learning cost relatively low to promote new ideas (but model could still be flawed) since
sds_lam = 4
means_lam = 25

# life of each scientist variable?

# for counting number of html pages generated / other useful counter value for debugging
count = 0