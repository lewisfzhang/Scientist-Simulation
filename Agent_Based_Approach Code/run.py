# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 14:05:55 2018

@author: zstemple
"""

# run.py
import os
os.chdir("C:/Users/zstemple/Documents/GitHub/Scientist-Simulation/Agent_Based_Approach Code")
from model import *

test_model = ScientistModel(N = 2, ideas_per_time = 1, time_periods = 3)
for i in range(5):   
    test_model.step()
