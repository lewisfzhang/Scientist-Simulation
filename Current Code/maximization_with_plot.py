from enum import Enum
import itertools
from possible_effort_splits import *
from classes import *
from optimization_equations import modified_normal_cdf

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class TimePeriod(Enum):
	tminusone = 0
	t = 1
	tplusone = 2

def return_for_old_young_pair(young_split_constant, old_split_constant, dict_of_pairs):
	max_return_young_old_pair = None
	max_return = 0

	constant_return = calculate_young_returns(young_split_constant, young_split_constant, old_split_constant) \
						+ calculate_old_returns(old_split_constant, young_split_constant, young_split_constant, old_split_constant)

	all_young_old_splits = []
	for young_split in dict_of_pairs.keys():
		all_old_splits = dict_of_pairs[young_split]
		for old_split in all_old_splits:
			all_young_old_splits.append((young_split, old_split))

	for young_old_pair in all_young_old_splits:
		# find return for young
		young_split = young_old_pair[0]

		young_return = calculate_young_returns(young_split, young_split_constant, old_split_constant)
		# find return for old
		old_split = young_old_pair[1]

		old_return = calculate_old_returns(old_split, young_split, young_split_constant, old_split_constant)

		# print "Old Pair:"
		# print max_return_young_old_pair
		# print max_return

		# if young_old_pair[1][0] == 0.0 and young_old_pair[1][1] != 0 and young_old_pair[1][2] == 0.0:
		# 	print young_old_pair
		# 	print young_return + old_return
		# if young_old_pair[1][0] != 0.0 and young_old_pair[1][1] != 0.0 and young_old_pair[1][2] == 0.0:
		# 	print young_old_pair
		# 	print young_return + old_return
		if young_return + old_return > max_return and constant_return < young_return + old_return:
			# print "Pair Return is greater:"
			# print young_old_pair
			# print young_return + old_return

			max_return_young_old_pair = young_old_pair
			max_return = young_return + old_return

		# else:
		# 	print "Pair Return NOT greater:"
		# 	print young_old_pair
		# 	print young_return + old_return

	return max_return_young_old_pair, max_return



def calculate_old_returns(old_split, young_split, young_split_constant, old_split_constant):
	idea_tminusone_effort = old_split[TimePeriod.tminusone]
	idea_t_effort = old_split[TimePeriod.t]
	idea_tplusone_effort = old_split[TimePeriod.tplusone]

	prev_effort_on_idea_t = young_split[TimePeriod.t] + old_split_constant[TimePeriod.tplusone]
	prev_effort_on_idea_tminusone = old_split_constant[TimePeriod.tplusone] + young_split_constant[TimePeriod.t] \
										+ old_split_constant[TimePeriod.t] + young_split[TimePeriod.tminusone]
	prev_effort_on_idea_tplusone = 0

	if has_n_zero_elems(old_split, 0):
		old_return = modified_normal_cdf(prev_effort_on_idea_tminusone + idea_tminusone_effort) \
						- modified_normal_cdf(prev_effort_on_idea_tminusone) \
						+ modified_normal_cdf(prev_effort_on_idea_t + idea_t_effort + young_split_constant[0]) \
						- modified_normal_cdf(prev_effort_on_idea_t) \
						+ modified_normal_cdf(prev_effort_on_idea_tplusone + idea_tplusone_effort + young_split_constant[1]) \
						- modified_normal_cdf(prev_effort_on_idea_tplusone)
	elif has_n_zero_elems(old_split, 1):
		if old_split[TimePeriod.tminusone] == 0:
			old_return = modified_normal_cdf(prev_effort_on_idea_t + idea_t_effort + young_split_constant[0]) \
						- modified_normal_cdf(prev_effort_on_idea_t) \
						+ modified_normal_cdf(prev_effort_on_idea_tplusone + idea_tplusone_effort + young_split_constant[1]) \
						- modified_normal_cdf(prev_effort_on_idea_tplusone)
		if old_split[TimePeriod.t] == 0:
			old_return = modified_normal_cdf(prev_effort_on_idea_tminusone + idea_tminusone_effort) \
						- modified_normal_cdf(prev_effort_on_idea_tminusone) \
						+ modified_normal_cdf(prev_effort_on_idea_tplusone + idea_tplusone_effort + young_split_constant[1]) \
						- modified_normal_cdf(prev_effort_on_idea_tplusone)
		if old_split[TimePeriod.tplusone] == 0:
			old_return = modified_normal_cdf(prev_effort_on_idea_tminusone + idea_tminusone_effort) \
						- modified_normal_cdf(prev_effort_on_idea_tminusone) \
						+ modified_normal_cdf(prev_effort_on_idea_t + idea_t_effort + young_split_constant[0]) \
						- modified_normal_cdf(prev_effort_on_idea_t) 
	elif has_n_zero_elems(old_split, 2):
		if old_split[0] == 0 and old_split[1] == 0:
			old_return = modified_normal_cdf(prev_effort_on_idea_tplusone + idea_tplusone_effort + young_split_constant[1]) \
						- modified_normal_cdf(prev_effort_on_idea_tplusone)
		if old_split[0] == 0 and old_split[2] == 0:
			old_return = modified_normal_cdf(prev_effort_on_idea_t + idea_t_effort + young_split_constant[0]) \
						- modified_normal_cdf(prev_effort_on_idea_t) 
		if old_split[1] == 0 and old_split[2] == 0:
			# print "prev_effort_on_idea_tminusone:"
			# print prev_effort_on_idea_tminusone

			# print "idea_tminusone_effort:"
			# print idea_tminusone_effort

			# print ' '
			old_return = modified_normal_cdf(prev_effort_on_idea_tminusone + idea_tminusone_effort) \
						- modified_normal_cdf(prev_effort_on_idea_tminusone)

	return old_return


def calculate_young_returns(young_split, young_split_constant, old_split_constant):
	idea_tminusone_effort = young_split[TimePeriod.tminusone]
	idea_t_effort = young_split[TimePeriod.t]

	prev_effort_on_idea_t = 0 
	prev_effort_on_idea_tminusone = young_split_constant[1] + old_split_constant[2]

	if young_split[0] == 0:
		young_return = modified_normal_cdf(idea_t_effort+prev_effort_on_idea_t+old_split_constant[2]) \
								 - modified_normal_cdf(prev_effort_on_idea_t)
	elif young_split[1] == 0:
		young_return = modified_normal_cdf(idea_tminusone_effort+prev_effort_on_idea_tminusone+old_split_constant[1]) \
								 - modified_normal_cdf(prev_effort_on_idea_tminusone)
	else:
		young_return = modified_normal_cdf(idea_tminusone_effort+prev_effort_on_idea_tminusone+old_split_constant[1]) \
								 - modified_normal_cdf(prev_effort_on_idea_tminusone)\
								 + modified_normal_cdf(idea_t_effort+prev_effort_on_idea_t+old_split_constant[2]) \
								 - modified_normal_cdf(prev_effort_on_idea_t)

	return young_return


def build_effort_pair_dict(young_splits, k, total_effort, size_of_effort_units, decimals):
	dict_of_pairs = {}
	for young_split in young_splits:
		dict_of_pairs[young_split] = all_possible_old_splits(young_split, k, total_effort, size_of_effort_units, decimals)
		print young_split
		print len(dict_of_pairs[young_split])

	return(dict_of_pairs)

def print_dict(dict_to_print):
	for item in dict_to_print.items():
		print(item)

# def modified_normal_cdf(x):
# 	return x**2

def main():

	# Set these for now: randomly select later?
	young_effort_constant = [0.4, 0.4]
	old_effort_constant = [0.3, 0.3, 0.3]


	size_of_effort_units = 0.01
	k = 0.1
	total_effort = 1.0
	decimals = 2

	# To make sure float calculations don't become a problem
	k = int(k*(10**decimals))
	total_effort = int(total_effort*(10**decimals))
	size_of_effort_units = int(size_of_effort_units*(10**decimals)) #Comment this more thoroughly because unintuitive


	young_splits = all_possible_young_splits(k, total_effort, size_of_effort_units, decimals)

	print "panda"

	# # set this for now: build function for this later
	# young_splits = [(0.4, 0.4), (0, 0.9), (0.9, 0), (0.3, 0.5), \
	# 								(0.5, 0.3), (0.1, 0.7), (0.7, 0.1), \
	# 								(0.6, 0.2), (0.2, 0.6)]

	possible_young_old_effort_pairs = build_effort_pair_dict(young_splits, k, total_effort, size_of_effort_units, decimals)	

	print "lion"
	print len(possible_young_old_effort_pairs)


	print sum(map(len, possible_young_old_effort_pairs.values()))

	print_dict(possible_young_old_effort_pairs)


	# Running the simulation:

	max_return_old_young_pair, max_return = return_for_old_young_pair(young_effort_constant, old_effort_constant, possible_young_old_effort_pairs)

	print "gorilla"

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = []
	y = []
	z = []
	a = []
	b = []

	x.append(max_return_old_young_pair[1][0])
	y.append(max_return_old_young_pair[1][1])
	z.append(max_return_old_young_pair[1][2])

	a.append(max_return_old_young_pair[0][0])
	b.append(max_return_old_young_pair[0][1])

	print max_return_old_young_pair
	print max_return
	counter = 1
	while young_effort_constant != max_return_old_young_pair[0] and old_effort_constant != max_return_old_young_pair[1] and counter <15:
		print "young effort constant"
		print young_effort_constant
		print "max_return_young"
		print max_return_old_young_pair[0]

		print "old effort constant"
		print old_effort_constant
		print "max_return_old"
		print max_return_old_young_pair[1]

		young_effort_constant = max_return_old_young_pair[0]
		old_effort_constant = max_return_old_young_pair[1]
		max_return_old_young_pair, max_return = return_for_old_young_pair(young_effort_constant, old_effort_constant, possible_young_old_effort_pairs)
		x.append(max_return_old_young_pair[1][0])
		y.append(max_return_old_young_pair[1][1])
		z.append(max_return_old_young_pair[1][2])

		a.append(max_return_old_young_pair[0][0])
		b.append(max_return_old_young_pair[0][1])

		counter += 1
		print counter


	print max_return_old_young_pair
	print max_return

	color_old = range(0, len(x), 1)

	color = []

	for item in color_old:
		color.append(item/100)

	ax.scatter(x, y, z, c= color, marker='o')

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show()

	plt.scatter(a,b, c = color)

	plt.show()


	




if __name__ == "__main__":
    main()
