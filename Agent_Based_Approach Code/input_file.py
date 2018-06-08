# input_file.py

# from random import randint
# randint(1000,9999) --> when necessary
# for randomization best practice
seed = 1234

# for optimizations
prop_invest_limit = 0.6  # works when <0.5?

# model variables
time_periods = 5
ideas_per_time = 3  # haven't implemented >1...
N = 2  # number of scientists per time period, for now just stick to 2
max_investment_lam = 50
true_sds_lam = 4
true_means_lam = 25

# agent constants
start_effort_lam = 25
start_effort_decay = 2
k_lam = 10
sds_lam = 4
means_lam = 25

# life of each scientist variable?