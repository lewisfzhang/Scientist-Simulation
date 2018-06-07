# input.py
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np

from numpy.random import poisson

# model
time_periods = 3
ideas_per_time = 1
N = 2
total_ideas = ideas_per_time*(time_periods+2)  # calculated as constant
max_investment = poisson(lam=10, size=total_ideas)
true_sds = poisson(4, size=total_ideas)
true_means = poisson(25, size=total_ideas)

# agent
start_effort = poisson(lam=10)
start_effort_decay = 1
k = poisson(lam=2, size=total_ideas)
sds = poisson(4, total_ideas)
means = poisson(25, total_ideas)
