from itertools import product
from enum import IntEnum
import sys
import cPickle as pickle
import numpy.ma as ma
import numpy as np

class TimePeriod(IntEnum):
	tminusone = 0
	t = 1
	tplusone = 2

def all_old_splits(total_effort, cost, effort_unit):
	options = range(0, total_effort+1, effort_unit)
	all_options = []

	for k in range(len(options)):
		for j in range(len(options)):
			for i in range(len(options)):
				effort_sum = options[i]+options[j]+options[k]
				if effort_sum == total_effort or effort_sum == (total_effort-cost) or effort_sum == (total_effort- (2*cost)):
					idec = options[i]
					jdec = options[j]
					kdec = options[k]
					all_options.append((idec, jdec, kdec))

	return all_options

def all_young_splits(total_effort, cost, effort_unit):
	options = range(0, total_effort+1, effort_unit)
	all_options = (product(options, options))
	return all_options


def remove_impossible_young_splits(all_young_effort_splits, total_effort, size_of_effort_units, k):
	splits_to_remove = set()
	for effort_split in all_young_effort_splits:
		if has_n_zero_elems(effort_split, 0) and sum(effort_split) != total_effort-2*k:
			splits_to_remove.add(effort_split)
		elif has_n_zero_elems(effort_split, 1) and sum(effort_split) != total_effort-k:
			splits_to_remove.add(effort_split)

	remove_from_collection(all_young_effort_splits, splits_to_remove)


def remove_impossible_old_splits(all_old_effort_splits, total_effort, size_of_effort_units, k):
	splits_to_remove = set()
	for effort_split in all_old_effort_splits:
		if effort_split[TimePeriod.tplusone] == total_effort:
			splits_to_remove.add(effort_split)
		# elif effort_split[TimePeriod.tplusone] != 0 and sum(effort_split) == total_effort:
		elif effort_split[TimePeriod.tplusone] != 0 and sum(effort_split) > (total_effort-k):
			splits_to_remove.add(effort_split)
		elif sum(effort_split) == total_effort-2*k and has_n_zero_elems(effort_split, 2):
			splits_to_remove.add(effort_split)
		elif effort_split[TimePeriod.tplusone] == 0 and sum(effort_split) == total_effort-2*k:
			splits_to_remove.add(effort_split)

	remove_from_collection(all_old_effort_splits, splits_to_remove)


# CAUTION: Call this before you make everything into a decimal to avoid float issues
def remove_old_splits_based_on_young_effort_splits(young_split, possible_old_splits, k, total_effort):

	splits_to_remove = set()

	# Case 1: if there is no zero in the splits
	if has_n_zero_elems(young_split, 0):
		for split in possible_old_splits:
			# if sum(split) == (total_effort-(2*k_young)):
			if sum(split) > (total_effort - k):
				splits_to_remove.add(split)
			elif split[TimePeriod.tplusone] == 0:
				if sum(split) != total_effort:
					splits_to_remove.add(split)
			elif split[TimePeriod.tplusone] != 0:
				if sum(split) != (total_effort-k):
					splits_to_remove.add(split)

	# Case 2: if the young scientist invested nothing in idea tminusone
	elif young_split[TimePeriod.tminusone] == 0:
		for split in possible_old_splits:
			if split[TimePeriod.tminusone] !=0 and split[TimePeriod.tplusone] != 0:
				if sum(split) != (total_effort-(2*k)):
					splits_to_remove.add(split)
			elif split[TimePeriod.tminusone] != 0 and split[TimePeriod.tplusone] == 0:
				if sum(split) != (total_effort-k):
					splits_to_remove.add(split)
			elif split[TimePeriod.tminusone] == 0 and split[TimePeriod.tplusone] != 0:
				if sum(split) != (total_effort-k):
					splits_to_remove.add(split)
			elif split[TimePeriod.tminusone] == 0 and split[TimePeriod.tplusone] == 0:
				if sum(split) != total_effort:
					splits_to_remove.add(split)

	# Case 3: If the young scientist invested nothing in idea t
	elif young_split[TimePeriod.t] == 0:
		for split in possible_old_splits:
			if split[TimePeriod.t] != 0 and split[TimePeriod.tplusone] != 0:
				if sum(split) != (total_effort-(2*k)):
					splits_to_remove.add(split)
			elif split[TimePeriod.t] != 0 and split[TimePeriod.tplusone] == 0:
				if sum(split) != (total_effort-k):
					splits_to_remove.add(split)
			elif split[TimePeriod.t] == 0 and split[TimePeriod.tplusone] != 0:
				if sum(split) != (total_effort-k):
					splits_to_remove.add(split)
			elif split[TimePeriod.t] == 0 and split[TimePeriod.tplusone] == 0:
				if sum(split) != total_effort:
					splits_to_remove.add(split)


	# print_collection_and_length(sort_list_of_tuples(splits_to_remove))
	remove_from_collection(possible_old_splits, splits_to_remove)


def sort_list_of_tuples(unsorted_list):
	return sorted(unsorted_list, key=lambda element: (element[0], element[1], element[2]))

# all possible cases:
# When young == (0, t)
# CASE 1: (t-1) and (t+1) != 0
# CASE 2: (t-1) != 0 and (t+1) == 0
# CASE 3: (t-1) == 0 and (t+1) != 0
# CASE 4: (t-1) == 0 and (t+1) ==0

# When young == (t, 0)
# CASE 1: (t) and (t+1) != 0
# CASE 2: (t) != 0 and (t+1) == 0
# CASE 3: (t) == 0 and (t+1) != 0
# CASE 4: t ==0 and (t+1) == 0

def print_collection_and_length(collection):
	for item in collection:
		print(item)
	print("Length: ")
	print(len(collection))
	print(' ')

# Only works for sets and lists
def remove_from_collection(original_collection, to_remove):
	for item in to_remove:
		original_collection.remove(item)


def has_n_zero_elems(list_of_nums, num_of_zeroes):
	zero_elems = [i for i, e in enumerate(list_of_nums) if e == 0]
	return len(zero_elems) == num_of_zeroes

def values_equal(val1, val2):
	return int(10*val1) == int(10*val2)


# Makes list of tuples into list of lists
def make_decimal(list_to_decimal, places):
	made_to_decimal = []
	for item in list_to_decimal:
		temp_item = list(item)
		for j in range(len(temp_item)):
			temp_item[j] = temp_item[j] / (10.0 ** places)
		made_to_decimal.append(tuple(temp_item))
	return made_to_decimal


def float_equals_int(float_num, int_num):
	if int(1-float_num) == int_num:
		return True
	return False


def all_possible_young_splits(young_splits, k, total_effort, size_of_effort_units, decimals):
	# young_splits = all_young_splits(total_effort, k, size_of_effort_units)
	remove_impossible_young_splits(young_splits, total_effort, size_of_effort_units, k)
	young_splits = make_decimal(young_splits, decimals)
	return young_splits


def all_possible_old_splits(old_splits, young_split, k_old, k_young, total_effort, size_of_effort_units, decimals):
	remove_impossible_old_splits(old_splits, total_effort, size_of_effort_units, k_old)
	remove_old_splits_based_on_young_effort_splits(young_split, old_splits, k_old, total_effort)
	old_splits = make_decimal(old_splits, decimals)
	return old_splits


def main(k, total_effort, size_of_effort_units, decimals, young_split1, young_split2):
	# k = float(raw_input("K? "))
	# total_effort = float(raw_input("Total efforts? "))
	# # size_of_effort_units = float(raw_input("Number of effort units?" ))
	# size_of_effort_units = float(raw_input("Size of effort unit?" ))
	# decimals = int(raw_input("Decimals? "))

	# young_split1 = float(raw_input("First value of young split?"))
	# young_split2 = float(raw_input("Second value of young split?"))

	# k = sys.argv[1]
	# total_effort = sys.argv[2]
	# size_of_effort_units = sys.argv[3]
	# decimals = sys.argv[4]
	# young_split1 = sys.argv[5]
	# young_split2 = sys.argv[6]

	# k = 0.1
	# total_effort = 1.0
	# size_of_effort_units = 0.1
	# decimals = 1
	# young_split1 = 0.0
	# young_split2 = 0.9

	young_split = [young_split1, young_split2]

	k = int(k*(10**decimals))
	total_effort = int(total_effort*(10**decimals))
	size_of_effort_units = int(size_of_effort_units*(10**decimals)) #Comment this more thoroughly because unintuitive

	all_old_effort_splits = all_old_splits(total_effort, k, size_of_effort_units)

	# print_collection_and_length(all_old_effort_splits)

	remove_impossible_old_splits(all_old_effort_splits, total_effort, size_of_effort_units, k)

	# print_collection_and_length(all_old_effort_splits)

	remove_old_splits_based_on_young_effort_splits(young_split, all_old_effort_splits, k, total_effort)

	all_old_effort_splits = make_decimal(all_old_effort_splits, decimals)

	# print_collection_and_length(sort_list_of_tuples(all_old_effort_splits))

	# all_young_effort_splits = all_young_splits(total_effort, k, size_of_effort_units)

	# remove_impossible_young_splits(all_young_effort_splits, total_effort, size_of_effort_units, k)

	# all_young_effort_splits = make_decimal(all_young_effort_splits, decimals)

	return all_old_effort_splits



if __name__ == "__main__":
    main()



