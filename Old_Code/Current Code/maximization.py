import csv
import sys
from enum import IntEnum
import itertools
from possible_effort_splits import *
from classes import *
from optimization_equations import modified_normal_cdf


class TimePeriod(IntEnum):
	tminusone = 0
	t = 1
	tplusone = 2

def return_for_young_old_pair(young_split_constant, old_split_constant, dict_of_pairs, std_dev):
	max_return_young_old_pair = (young_split_constant, old_split_constant)
	max_return = 0
	old_return_final = 0
	young_return_final = 0

	constant_return = calculate_young_returns(young_split_constant, young_split_constant, old_split_constant, std_dev) \
						+ calculate_old_returns(old_split_constant, young_split_constant, young_split_constant, old_split_constant, std_dev)
	all_young_old_splits = []
	for young_split in dict_of_pairs.keys():
		all_old_splits = dict_of_pairs[young_split]
		for old_split in all_old_splits:
			all_young_old_splits.append((young_split, old_split))

	for young_old_pair in all_young_old_splits:
		# find return for young
		young_split = young_old_pair[0]

		young_return = calculate_young_returns(young_split, young_split_constant, old_split_constant, std_dev)
		# find return for old
		old_split = young_old_pair[1]

		old_return = calculate_old_returns(old_split, young_split, young_split_constant, old_split_constant, std_dev)
		if young_return + old_return > max_return and constant_return < young_return + old_return:
			max_return_young_old_pair = young_old_pair
			max_return = young_return + old_return
			old_return_final = old_return
			young_return_final = young_return


	return max_return_young_old_pair, max_return, old_return_final, young_return_final



def calculate_old_returns(old_split, young_split, young_split_constant, old_split_constant, std_dev):
	idea_tminusone_effort = old_split[TimePeriod.tminusone]
	idea_t_effort = old_split[TimePeriod.t]
	idea_tplusone_effort = old_split[TimePeriod.tplusone]

	prev_effort_on_idea_t = young_split[TimePeriod.t] + old_split_constant[TimePeriod.tplusone]
	prev_effort_on_idea_tminusone = old_split_constant[TimePeriod.tplusone] + young_split_constant[TimePeriod.t] \
										+ old_split_constant[TimePeriod.t] + young_split[TimePeriod.tminusone]
	prev_effort_on_idea_tplusone = 0

	if has_n_zero_elems(old_split, 0):
		old_return = modified_normal_cdf(std_dev, prev_effort_on_idea_tminusone + idea_tminusone_effort) \
						- modified_normal_cdf(std_dev, prev_effort_on_idea_tminusone) \
						+ modified_normal_cdf(std_dev, prev_effort_on_idea_t + idea_t_effort + young_split_constant[0]) \
						- modified_normal_cdf(std_dev, prev_effort_on_idea_t) \
						+ modified_normal_cdf(std_dev, prev_effort_on_idea_tplusone + idea_tplusone_effort + young_split_constant[1]) \
						- modified_normal_cdf(std_dev, prev_effort_on_idea_tplusone)
	elif has_n_zero_elems(old_split, 1):
		if old_split[TimePeriod.tminusone] == 0:
			old_return = modified_normal_cdf(std_dev, prev_effort_on_idea_t + idea_t_effort + young_split_constant[0]) \
						- modified_normal_cdf(std_dev, prev_effort_on_idea_t) \
						+ modified_normal_cdf(std_dev, prev_effort_on_idea_tplusone + idea_tplusone_effort + young_split_constant[1]) \
						- modified_normal_cdf(std_dev, prev_effort_on_idea_tplusone)
		if old_split[TimePeriod.t] == 0:
			old_return = modified_normal_cdf(std_dev, prev_effort_on_idea_tminusone + idea_tminusone_effort) \
						- modified_normal_cdf(std_dev, prev_effort_on_idea_tminusone) \
						+ modified_normal_cdf(std_dev, prev_effort_on_idea_tplusone + idea_tplusone_effort + young_split_constant[1]) \
						- modified_normal_cdf(std_dev, prev_effort_on_idea_tplusone)
		if old_split[TimePeriod.tplusone] == 0:
			old_return = modified_normal_cdf(std_dev, prev_effort_on_idea_tminusone + idea_tminusone_effort) \
						- modified_normal_cdf(std_dev, prev_effort_on_idea_tminusone) \
						+ modified_normal_cdf(std_dev, prev_effort_on_idea_t + idea_t_effort + young_split_constant[0]) \
						- modified_normal_cdf(std_dev, prev_effort_on_idea_t) 
	elif has_n_zero_elems(old_split, 2):
		if old_split[0] == 0 and old_split[1] == 0:
			old_return = modified_normal_cdf(std_dev, prev_effort_on_idea_tplusone + idea_tplusone_effort + young_split_constant[1]) \
						- modified_normal_cdf(std_dev, prev_effort_on_idea_tplusone)
		if old_split[0] == 0 and old_split[2] == 0:
			old_return = modified_normal_cdf(std_dev, prev_effort_on_idea_t + idea_t_effort + young_split_constant[0]) \
						- modified_normal_cdf(std_dev, prev_effort_on_idea_t) 
		if old_split[1] == 0 and old_split[2] == 0:
			old_return = modified_normal_cdf(std_dev, prev_effort_on_idea_tminusone + idea_tminusone_effort) \
						- modified_normal_cdf(std_dev, prev_effort_on_idea_tminusone)

	return old_return


def calculate_young_returns(young_split, young_split_constant, old_split_constant, std_dev):
	idea_tminusone_effort = young_split[TimePeriod.tminusone]
	idea_t_effort = young_split[TimePeriod.t]

	prev_effort_on_idea_t = 0 
	prev_effort_on_idea_tminusone = young_split_constant[1] + old_split_constant[2]

	if young_split[0] == 0:
		young_return = modified_normal_cdf(std_dev, idea_t_effort+prev_effort_on_idea_t+old_split_constant[2]) \
								 - modified_normal_cdf(std_dev, prev_effort_on_idea_t)
	elif young_split[1] == 0:
		young_return = modified_normal_cdf(std_dev, idea_tminusone_effort+prev_effort_on_idea_tminusone+old_split_constant[1]) \
								 - modified_normal_cdf(std_dev, prev_effort_on_idea_tminusone)
	else:
		young_return = modified_normal_cdf(std_dev, idea_tminusone_effort+prev_effort_on_idea_tminusone+old_split_constant[1]) \
								 - modified_normal_cdf(std_dev, prev_effort_on_idea_tminusone)\
								 + modified_normal_cdf(std_dev, idea_t_effort+prev_effort_on_idea_t+old_split_constant[2]) \
								 - modified_normal_cdf(std_dev, prev_effort_on_idea_t)

	return young_return


def build_effort_pair_dict(young_splits, k_old, k_young, total_effort, size_of_effort_units, decimals):
	old_splits = all_old_splits(total_effort, k_old, size_of_effort_units)
	dict_of_pairs = {}
	# young_split = (9, 0)
	# all_old = all_possible_old_splits(old_splits, young_split, k_old, k_young, total_effort, size_of_effort_units, decimals)
	for young_split in young_splits: 
		# remember that [:] makes a copy of a list
		dict_of_pairs[young_split] = all_possible_old_splits(old_splits[:], young_split, k_old, k_young, total_effort, size_of_effort_units, decimals)
	return(dict_of_pairs)

def print_dict(dict_to_print):
	for item in dict_to_print.items():
		print(item)

def main():
	
	args = [float(x) for x in sys.argv[1:]]
	
	size_of_effort_units = args[0]
	float_k_young = args[1]
	float_k_old = args[2]
	total_effort = args[3]
	decimals = args[4]
	young_effort_constant = (args[5], args[6])
	old_effort_constant = (args[7], args[8], args[9])
	reps = args[10]
	std_dev = args[11]
	# Set these for now: randomly select later?
#	young_effort_constant = (0.3, 0.3)
#	old_effort_constant = (0.1, 0.1, 0.6)
# 
#	size_of_effort_units = 0.1
#	k = 0.1
#	total_effort = 1.0
#	decimals = 2

	# To make sure float calculations don't become a problem
	k_young = int(float_k_young*(10**decimals))
	k_old = int(float_k_old*(10**decimals))
	total_effort = int(total_effort*(10**decimals))
	size_of_effort_units = int(size_of_effort_units*(10**decimals)) #Comment this more thoroughly because unintuitive

	young_splits = all_young_splits(total_effort, k_young, size_of_effort_units)
	young_splits = all_possible_young_splits(young_splits, k_young, total_effort, size_of_effort_units, decimals)

	possible_young_old_effort_pairs = build_effort_pair_dict(young_splits, k_old, k_young, total_effort, size_of_effort_units, decimals)	

	# print "----------------------------------------------------DONE BUILDING DICTIONARY----------------------------------------------------"
	
	# print_dict(possible_young_old_effort_pairs)
	# Running the simulation:
	to_write_rows = []

	for rep in range(0, int(reps)):
		max_return_old_young_pair, max_return, old_return, young_return = return_for_young_old_pair(young_effort_constant, old_effort_constant, possible_young_old_effort_pairs, std_dev)

		young_effort_constant = max_return_old_young_pair[0]
		old_effort_constant = max_return_old_young_pair[1]
		
		to_write = [float_k_young, float_k_old, round(young_return, 3), round(old_return, 3), round(max_return,3), max_return_old_young_pair[0][0], max_return_old_young_pair[0][1], \
						max_return_old_young_pair[1][0], max_return_old_young_pair[1][1], max_return_old_young_pair[1][2]]
		to_write_str = [str(x) for x in to_write]
		to_write_rows.append(to_write_str)

	with open('test_young_' + str(float_k_young) + '_old_' + str(float_k_old) +'.csv', 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter = ',')
		for row in to_write_rows:
			writer.writerow(row)



	




if __name__ == "__main__":
    main()
