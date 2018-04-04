#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 12:59:46 2017

@author: michellezhao
"""

from model import *
import matplotlib.pyplot as plt


model = ScientistModel(2, 1, 5)
for i in range(10):
   model.step()

# all_wealth = []
# for j in range(100):
#     model = ScientistModel(10, 2, 10)
#     for i in range(10):
#         model.step()
#     for agent in model.schedule.agents:
#         all_wealth.append(agent.wealth)
    
# plt.hist(all_wealth, bins=range(max(all_wealth)+1))