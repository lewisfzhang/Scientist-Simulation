

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def main():
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	

	x = [.89,  0.0, .87, 0.0, 0.0, .87, 0.0, 0.0]
	y = [0.0, 1.0, .13, 1.0, 0.0, .13, 1.0, 0.0]
	z = [0.01, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.9]
	
	color_old = range(0, len(x), 1)

	color = []

	for item in color_old:
		color.append(item/len(x))

	ax.plot(x, y, z, c= color, marker='o')

	ax.set_xlabel('Idea t-1')
	ax.set_ylabel('Idea t')
	ax.set_zlabel('Idea t+1')

	plt.show()

	a = [.9, 0.0, .9, .48, 0.0, .9, .48, 0.0, .9]
	b = [0.0, .9, 0.0, .32, .9, 0.0, .32, .9, 0.0]


	plt.plot(a,b, c = color)
	plt.xlabel('Idea t-1')
	plt.ylabel('Idea t')
	
	plt.show()

main()




