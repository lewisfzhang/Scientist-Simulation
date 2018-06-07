# input.py
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np

from numpy.random import poisson

# model variables
time_periods = 10
ideas_per_time = 1
N = 2
max_investment_lam = 10
true_sds_lam = 4
true_means_lam = 25

# # model poisson curves
# max_investment = poisson(lam=max_investment_lam, size=total_ideas)
# true_sds = poisson(lam=true_sds_lam, size=total_ideas)
# true_means = poisson(lam=true_means_lam, size=total_ideas)


# agent constants
start_effort_lam = 10
start_effort_decay = 1
k_lam = 2
sds_lam = 4
means_lam = 25

# agent variables
# start_effort = poisson(lam=start_effort_lam)
# start_effort_decay = 1
# k = poisson(lam=k_lam, size=total_ideas)
# sds = poisson(lam=sds_lam, size=total_ideas)
# means = poisson(lam=sds_lam, size=total_ideas)
