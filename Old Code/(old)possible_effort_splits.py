import itertools

# Function: all_possible_effort_splits
# Parameters: 
# num_of_effort_units - the number of effort units the total K efforts is split into (i.e. 10 means K/10 is one effort unit)
# number_of_ideas - number of ways we want to split the effort units
# Given a list, gives you all possible combinations of effort 
# def all_possible_effort_splits(total_effort, num_of_effort_units, num_of_ideas, k):
# 	effort_unit = float(total_effort)/num_of_effort_units
# 	rng = list(range(num_of_effort_units+1))*num_of_ideas
# 	rng = [round(x * effort_unit, 2) for x in rng]
# 	effort_splits = set(i for i in itertools.permutations(rng, num_of_ideas) if values_equal(sum(i), total_effort))
# 	effort_splits = effort_splits.union(i for i in itertools.permutations(rng, num_of_ideas) if values_equal(sum(i), total_effort-k))
# 	effort_splits = effort_splits.union(i for i in itertools.permutations(rng, num_of_ideas) if values_equal(sum(i), total_effort-2*k))
# 	return(effort_splits)



def all_possible_effort_splits():
	#options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	options = range(11)
	list_of_options = []
	

	for i in range(len(options)):
		for j in range(len(options)):
			for k in range(len(options)):
				if (i+j+k) == 10 or (i+j+k) == 9 or (i+j+k) == 8:
					idec = i/10.0
					jdec = j/10.0
					kdec = k/10.0
					list_of_options.append((idec, jdec, kdec))

	print len(list_of_options)

	return set(list_of_options)
					

def remove_impossible_splits(all_effort_splits, total_effort, num_of_effort_units, num_of_ideas, k):
	splits_to_remove = set()
	for effort_split in all_effort_splits:
		# print(effort_split)
		# print(effort_split[0])
		# print(sum(effort_split))
		# print(effort_split[0] != 0.0 and sum(effort_split) == total_effort)

		if values_equal(effort_split[0], total_effort): # Case 1
			splits_to_remove.add(effort_split)
		elif not values_equal(effort_split[0], 0.0) and values_equal(sum(effort_split), total_effort):
		# elif int(1000*effort_split[0]) != int(1000*0.0) and values_equal(sum(effort_split), total_effort):
			splits_to_remove.add(effort_split)
		elif values_equal(sum(effort_split), total_effort-2*k) and two_zero_elem(effort_split):
			splits_to_remove.add(effort_split)
		elif values_equal(effort_split[0], 0.0) and values_equal(sum(effort_split), total_effort-2*k):
			splits_to_remove.add(effort_split)

	all_effort_splits = all_effort_splits.difference(splits_to_remove)
	return all_effort_splits


def two_zero_elem(list_of_nums):
	zero_elems = [i for i, e in enumerate(list_of_nums) if e == 0]
	return len(zero_elems) == 2

def values_equal(val1, val2):
	return int(10*val1) == int(10*val2)



# def print_sorted_set(unsorted_set):
# 	sorted_list = list(unsorted_set)
# 	print(sorted_list)
# 	sorted_list = sorted_list.sort()
# 	print(sorted_list)

def float_equals_int(float_num, int_num):
	if int(1-float_num) == int_num:
		return True
	return False

def main():
	k = 0.1
	total_effort = 1.0
	num_of_effort_units = 10 #Comment this more thoroughly because unintuitive
	num_of_ideas = 3
	# all_effort_splits = all_possible_effort_splits(total_effort, num_of_effort_units, num_of_ideas, k)
	# print((0.2, 0.0, 0.4) in all_effort_splits)
	# print(all_effort_splits)
	all_effort_splits = all_possible_effort_splits()

	for effort_split in all_effort_splits:
		if values_equal(effort_split[0], 0.6):
			print effort_split

	all_effort_splits = remove_impossible_splits(all_effort_splits, total_effort, num_of_effort_units, num_of_ideas, k)

	print(all_effort_splits)
	print(len(all_effort_splits))

	list_of_interest = []
	for effort_split in all_effort_splits:
		if values_equal(effort_split[0], 0.0):
			list_of_interest.append(effort_split)

	print len(list_of_interest)
	print(list_of_interest)
	# print_sorted_set(all_effort_splits)

if __name__ == "__main__":
    main()


#int(1 - sum(effort_split)) == 0)

