#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:16:27 2017

@author: michellezhao
"""

from scipy.stats import logistic
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(1,1)

x = np.linspace(0,2000)
ax.plot(x, logistic.cdf(x, loc = 500, scale = 100), 'r-', lw=5, alpha=.1, label= 'gompertz cdf')
plt.show()
# print("50",logistic.cdf(50,loc=25,scale=4))
# print("70",logistic.cdf(60,loc=25,scale=4))