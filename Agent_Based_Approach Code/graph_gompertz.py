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


x = np.linspace(0,50)
ax.plot(x, logistic.cdf(x, loc = 3, scale = 1), 'r-', lw=5, alpha=0.6, label= 'gompertz cdf')
plt.show()