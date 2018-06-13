import numpy as np
import pylab
from scipy.optimize import curve_fit

def sigmoid(x, x0, k, a, c):
     y = a / (1 + np.exp(-k*(x-x0))) + c
     return y

xdata = np.array([0.0,   1.0,  3.0,  4.3,  7.0,   8.0,   8.5, 10.0,  
12.0, 14.0])
ydata = np.array([0.11, 0.12, 0.14, 0.21, 0.83,  1.45,   1.78,  1.9, 
1.98, 2.02])

popt, pcov = curve_fit(sigmoid, xdata, ydata)
print "Fit:"
print "x0 =", popt[0]
print "k  =", popt[1]
print "a  =", popt[2]
print "c  =", popt[3]
print "Asymptotes are", popt[3], "and", popt[3] + popt[2]

x = np.linspace(-1, 15, 50)
y = sigmoid(x, *popt)


pylab.plot(xdata, ydata, 'o', label='data')
pylab.plot(x,y, label='fit')
pylab.ylim(0, 2.05)
pylab.legend(loc='upper left')
pylab.grid(True)
pylab.show()