# functions.py

import numpy as np
import config
import timeit
import glob
import os
# import resource
# from pympler import asizeof
import gc
import ast


# Input: Parameters for the logistic cumulative distribution function
# Output: Value at x of the logistic cdf defined by the location and scale parameter
def logistic_cdf(x, loc, scale):
    return 1/(1+np.exp((loc-x)/scale))


# round to 2 decimal places and returns the immutable tuple object for datacollector
def rounded_tuple(array):
    return tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, tuple(array)))


# Input:
# 1) num_ideas (scalar): number of ideas to create the return matrix for
# 2) max_of_max_inv (scalar): the maximum of all the maximum investment limits over all ideas
# 3) sds (array): the standard deviation parameters of the return curve of each idea
# 4) means (array): the mean parameters of the return curve of each idea
# Output:
# A matrix that is has dimension n x m where n = num_ideas and m = max_of_max_inv
# where each cell (i,j) contains the return based on the logistic cumulative
# distribution function of the i-th idea and the j-th extra unit of effort
def create_return_matrix(num_ideas, sds, means, M, sds_lam, means_lam):
    # 4 std dev outside normal curve basically guarentees all values
    x = np.arange(means_lam*2+1)
    returns_list = []
    for i in range(num_ideas):
        # Calculates the return for an idea for all amounts of effort units
        returns = M[i] * logistic_cdf(x, loc=means[i], scale=sds[i])
        # Stacks arrays horizontally
        to_subt_temp = np.hstack((0, returns[:-1]))
        # Calculates return per unit of effort
        returns = returns - to_subt_temp
        returns_list.append(returns)
    return np.array(returns_list)


def get_returns(idea, returns_info, start_idx, end_idx):
    M = returns_info[0][idea]
    sds = returns_info[1][idea]
    means = returns_info[2][idea]
    start = M * logistic_cdf(start_idx, loc=means, scale=sds)
    end = M * logistic_cdf(end_idx, loc=means, scale=sds)
    return end-start


# for counting number of html pages generated
def page_counter():
    config.count += 1
    return config.count


def reset_counter():
    config.count = 0


def flatten_list(list_name):
    return_list = []
    for x in range(len(list_name)):
        for idx, val in enumerate(list_name[x]):
            return_list.append(val)
    return return_list


# helper method for calculating runtime
def stop_run(string):
    print("")
    print(string)
    # end runtime
    stop = timeit.default_timer()
    print("Elapsed runtime: ", stop - config.start, "seconds")
    config.start = stop


# np.log() that handles 0 (and very small values that will return infinity)
# CONDITION: np_array cannot be a list, must be a np array
def log_0(np_array):
    return np.log(np_array, out=np.zeros(len(np_array)), where=np_array > 2**-10)


# np.divide that handles division by 0
# num, denom can also be lists!
def divide_0(num, denom):
    return np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)


def png_to_html():
    image_list = glob.glob('web/images/*.png')
    html = ''
    for i in range(len(image_list)):
        html += '<img src="'+str(os.getcwd())+'/'+str(image_list[i])+'" />'
    with open("web/pages/all_images.html", "w") as file:
        file.write(html)


# returns current resources used
# def mem():
#     print('Memory usage         : % 2.2f MB' % round(
#         resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0, 1)
#     )


# returns size of object
# def get_size(obj):
#     return asizeof.asizeof(obj)


# create directory in tmp if it doesn't already exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def gc_collect():
    gc.collect()
    # print("\nBEFORE", end="")
    # mem()
    # collected = gc.collect()
    # print("Collected", collected, "objects")
    # print("AFTER", end="")
    # mem()


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def df_formatter(array, title):
    s = ''
    for idx, val in enumerate(array):
        if val != 0:
            s += "{'idea': "+str(idx)+", '"+title+"': "+str(round(val, 2))+"}\r\n"
    if s == '':
        s = '-'
    return s


def str_to_dict(s):
    return ast.literal_eval(s)
