from math import e
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# def sigmoid(r, x):
# 	return 1.0/(1+e**(-(r*x)))


# def modified_sigmoid(r, z, x):
# 	return (sigmoid(r, x+z) - sigmoid(r, z))/(1-sigmoid(r, z))

#def modified_sigmoid(r, x, z):
# 	return sigmoid(r, x+z)

# def graph_sigmoid(r, x_min, x_max, num_of_intervals):
# 	x_range = np.linspace(x_min, x_max, num_of_intervals)
# 	y = []
# 	for x in x_range:
# 		y.append(norm.cdf(x))
# 	plt.plot(x_range,y)
# 	plt.show()

def graph_normal_cdf(x_min, x_max, num_of_intervals, mean, std_dev):
	x_range = np.linspace(x_min, x_max, num_of_intervals)
	y = []
	for x in x_range:
		y.append(norm.cdf(x, mean, std_dev))
	plt.plot(x_range,y)
	plt.show()

# def graph_modified_sigmoid(r, z, x_min, x_max, num_of_intervals):
# 	x_range = np.linspace(x_min, x_max, num_of_intervals)
# 	y = []
# 	for x in x_range:
# 		y.append(modified_sigmoid(r, z, x))
# 	plt.plot(x_range, y)
# 	plt.show()

def graph_modified_normal_cdf(x_min, x_max, num_of_intervals, mean, std_dev, horizontal_shift, vertical_shift):
	x_range = np.linspace(x_min, x_max, num_of_intervals)
	y = []
	for x in x_range:
		y.append(norm.cdf(x-horizontal_shift, mean, std_dev)-vertical_shift)
	plt.plot(x_range,y)
	plt.show()

def modified_normal_cdf(std_dev, x):
	return (norm.cdf(x-2, 0, std_dev)-0.0478)

def main():
	# graph_sigmoid(1, -5, 5, 50)
	# graph_modified_sigmoid(2.0, , -5, 5, 15)
	# print norm.cdf(1)

	graph_modified_normal_cdf(0, 10, 50, 0, 1.2, 4, 0)


if __name__ == "__main__":
    main()
