#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:45:51 2017

@author: michellezhao
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp

# Plotting logistic CDF with different s and mu
loc, scale = 100, 1
#s = np.random.logistic(loc, scale, 10000)
#count, bins, ignored = plt.hist(s, bins=50)


def logistic_cdf(x, loc, scale):
    return 1/(1+exp((loc-x)/scale))

def gompertz_cdf ( x, a, b ):

#*****************************************************************************
#
## GOMPERTZ_CDF evaluates the Gompertz CDF.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    03 April 2016
#
#  Author:
#
#    John Burkardt
#
#  Reference:
#
#    Johnson, Kotz, and Balakrishnan,
#    Continuous Univariate Distributions, Volume 2, second edition,
#    Wiley, 1994, pages 25-26.
#
#  Parameters:
#
#    Input, real X, the argument of the CDF.
#
#    Input, real A, B, the parameters of the PDF.
#    1 < A, 0 < B.
#
#    Output, real CDF, the value of the CDF.
#
  import numpy as np
  cdf = 1.0 - np.exp ( - b * ( a ** x - 1.0 ) / np.log ( a ) )

  return cdf

#plt.plot(logistic_cdf(bins, loc, scale)*count.max()/gompertz_cdf(bins, loc, scale).max())
#
#
#plt.show()





import pylab as p

def boltzman(x, xmid, tau):
    """
    evaluate the boltzman function with midpoint xmid and time constant tau
    over x
    """
    return 1. / (1. + np.exp(-(x-xmid)/tau))

x = np.arange(0, 20, .01)
S = boltzman(x, 10, 1)
Z = 1-boltzman(x, 10, 1)
p.plot(x, S, x, Z, color='red', lw=2)
p.axis([0, 20, 0, 1])
p.show()
