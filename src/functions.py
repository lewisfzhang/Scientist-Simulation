# functions.py

import numpy as np
import config, timeit, glob, os, gc, ast, random, math
import subprocess as s
from scipy.stats import cauchy


# Input: Parameters for the logistic cumulative distribution function
# Output: Value at x of the logistic cdf defined by the location and scale parameter
def old_logistic_cdf(x, loc, scale):
    return 1/(1+np.exp((loc-x)/scale))


# take the first derivative of logistic cdf
def old_logistic_cdf_derivative(x, loc, scale):
    return np.exp((loc - x)/scale)/(scale*(np.exp((loc - x)/scale) + 1)**2)


# remember anything from your calculus class?
def old_logistic_cdf_inv_deriv(slope_val, loc, scale):
    output = []
    for i in range(len(loc)):
        # np.roots takes coefficients of the polynomial
        r = np.roots([slope_val[i] * scale[i], slope_val[i] * scale[i] * 2 - 1, slope_val[i] * scale[i]])
        r = r[np.isreal(r)]
        if len(r) == 0:  # no real roots
            raise Exception('No real roots when solving logistic cdf slope!')
        elif len(r) == 1:  # only one real root? check the program again
            raise Exception('Only one real root when solving logistic cdf slope!')
        else:
            output.append(loc[i] - np.log(r) * scale[i])
    return np.asarray(output)


# Input: Parameters for the logistic cumulative distribution function
# Output: Value at x of the logistic cdf defined by the location and scale parameter
def logistic_cdf(x, loc, scale):
    return (old_logistic_cdf(x, loc, scale) - old_logistic_cdf(0, loc, scale)) / (1 - old_logistic_cdf(0, loc, scale))


# given slope find location
def inv_logistic_cdf(y, loc, scale):
    return loc - scale * np.log(((1 - old_logistic_cdf(0, loc, scale)) * y + old_logistic_cdf(0, loc, scale)) ** -1 - 1)


# take the first derivative of logistic cdf
def logistic_cdf_derivative(x, loc, scale):
    return old_logistic_cdf_derivative(x, loc, scale) / (1 - old_logistic_cdf(0, loc, scale))


# too much work to calculate second derivative
def logistic_cdf_2d(x, idea, returns_info):
    sds = returns_info[1][idea]
    means = returns_info[2][idea]
    shift = returns_info[3][idea]
    return (logistic_cdf_derivative(x-shift+0.000001, loc=means, scale=sds) - logistic_cdf_derivative(x-shift-0.000001, loc=means, scale=sds)) / 0.000002


# probably not needed since idea quality can be found using simply M? ask jay...
def max_cdf_calc(returns_info):
    out = np.zeros(len(returns_info[0]))
    x = 300  # just random edgy number
    for idea in range(len(out)):
        M = returns_info[0][idea]
        sds = returns_info[1][idea]
        means = returns_info[2][idea]
        shift = returns_info[3][idea]
        out[idea] = M * logistic_cdf(x-shift, loc=means, scale=sds)
    return out


# remember anything from your calculus class?
def logistic_cdf_inv_deriv(slope_val, loc, scale):
    # strictly because too lazy to correctly update this formula, but it works based on new formula
    slope_val *= (1 - old_logistic_cdf(0, loc, scale))
    output = []
    for i in range(len(loc)):
        # np.roots takes coefficients of the polynomial
        r = np.roots([slope_val[i] * scale[i], slope_val[i] * scale[i] * 2 - 1, slope_val[i] * scale[i]])
        r = r[np.isreal(r)]
        if len(r) == 0:  # no real roots
            raise Exception('No real roots when solving logistic cdf slope!')
        elif len(r) == 1:  # only one real root? check the program again
            raise Exception('Only one real root when solving logistic cdf slope!')
        else:
            output.append(loc[i] - np.log(r) * scale[i])
    return np.asarray(output)


# round to 2 decimal places and returns the immutable tuple object for datacollector
def rounded_tuple(array):
    return tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, tuple(array)))


def get_returns(idea, returns_info, start_idx, end_idx):
    m = returns_info[0][idea]
    sds = returns_info[1][idea]
    means = returns_info[2][idea]
    shift = returns_info[3][idea]
    # WE DON'T NEED SHIFT RIGHT?!
    start = m * logistic_cdf(start_idx, loc=means, scale=sds)
    end = m * logistic_cdf(end_idx, loc=means, scale=sds)
    return end-start


# for counting number of html pages generated
def counter():
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
    f_print("")
    f_print(string)
    # end runtime
    stop = timeit.default_timer()
    f_print("Elapsed runtime: ", stop - config.start, "seconds")
    config.start = stop


# np.log() that handles 0 (and very small values that will return infinity)
# CONDITION: np_array cannot be a list, must be a np array
def log_0(np_array):
    return np.log(np_array, out=np.zeros(len(np_array)), where=np_array > 2**-10)


# np.divide that handles division by 0
# num, denom can also be lists!
def divide_0(num, denom):
    return np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)


def clear_images():
    s.call('rm '+config.parent_dir+'data/images/*.png', shell=True)


def png_to_html(tmp_path):
    if tmp_path is None:
        image_list = glob.glob(config.parent_dir + 'data/images/*.png')
        html = ''
        for path in image_list:
            html += '<img src="../' + str(path)[str(path).find('images'):] + '" />'
        with open(config.parent_dir + "data/pages/all_images.html", "w") as file:
            file.write(html)
    else:
        image_list = glob.glob(tmp_path + '*.png')
        html = ''
        for path in image_list:
            path = str(path)
            path = path[path.find('step/') + len('step/'):]
            path = path[path.find('/') + len('/'):]
            html += '<img src="' + path + '" />'
        with open(tmp_path + "all_images.html", "w") as file:
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


# PMF implementation that models Bayes' method/theorem P(A | B) = P(B | A) * P(A) / P(B)
# data is the ratio of over/underestimating in the past for the scientist
#
# # scientists initially hypothesize that they have an equal chance of over/underestimating returns
# # P(m > M)
# # m is scientist believed impact, M is the actual max impact of the idea
# pmf.Set('m > M', 0.5)
# pmf.Set('m <= M', 0.5)
#
# # scientists adjust their chances based on past investing data (across all ideas)
# # P(I | m > M) where I is slope
# pmf.Mult('m > M', data[0])
# pmf.Mult('m <= M', data[1])
#
# # same as dividing by P(I)
# pmf.Normalize()
def get_bayesian_formula(data):
    # flipped p0 and p1? not sure and check with jay
    p0 = 0.5 * data[1]
    p1 = 0.5 * data[0]
    return p0 / (p0 + p1)


# helper function that generates noise for each agent in init
def random_noise(seed1, seed2, unique_id, total_ideas, x):
    random.seed(config.seed_array[unique_id][seed1])
    sds = random.choice(np.arange(1, end_limit(x)))

    # ARRAY: the 'noise' or error based on each specific scientist
    # np.random.seed(config.seed_array[unique_id][seed2])
    # noise = np.random.normal(0, sds, total_ideas)  # void

    # trying cauchy distribution with wider tails compared to normal distribution
    # np.random.seed(config.seed_array[unique_id][seed2])
    # noise = cauchy.rvs(loc=0, scale=sds, size=total_ideas)

    # uniform is better?
    np.random.seed(config.seed_array[unique_id][seed2])
    # should be 3 but we don't want negative values for now
    noise = np.random.uniform(-3 * end_limit(x), 3 * end_limit(x), size=total_ideas)

    return noise


# protects against noise that produces negative sds/means
def end_limit(x):
    return int((x - 3 * math.sqrt(x)) / 3)


# returns proportion based on array of numbers
def get_pdf(arr):
    if arr.shape[0] == 1:
        return arr / sum(arr)
    else:
        out = []
        out.append(arr[0] / sum(arr[0]))
        out.append(arr[1] / sum(arr[1]))
        return np.asarray(out)


# returns cdf based on pdf array
def get_cdf(arr):
    return np.cumsum(get_pdf(arr))


def f_print(*s):
    out = ''
    for i in s:
        out += str(i) + ' '
    print(out)
    with open(config.parent_dir + 'data/output.txt', 'a') as f:
        f.write(out+'\n')


# note: written for big_data only
def process_dict(s):
    # guard
    if s != '-':
        new_list = []
        last_bracket = 0
        for i in range(s.count('idea')):
            left_bracket = s[last_bracket:].index('{')
            right_bracket = s[last_bracket:].index('}') + 1
            new_list.append(str_to_dict(s[last_bracket:][left_bracket:right_bracket]))
            last_bracket += right_bracket
        return new_list
    else:
        return []  # empty list exists out of the for loop


# helper function for final_slopes list format flattening
def flat_2d(l):  # l = slope
    num = len(l[0])  # num = num scientists
    order_idx = flatten_list(l[len(l) - 1])  # same as l[2]?
    new_list = [[] for a in range(num)]
    count = [[0, 0] for a in range(num)]
    for i in range(len(order_idx)):
        idx_id = order_idx[i][0]
        sci_id = order_idx[i][1] - 1  # 0 based index
        new_list[sci_id].append(l[idx_id][sci_id][count[sci_id][idx_id]])
        count[sci_id][idx_id] += 1
    return new_list


def expand_2d(l1, l2, l3):  # l2 = idea_idx, l3 = scientist_id
    l2 = rounded_tuple(flatten_list(l2))
    l3 = rounded_tuple(flatten_list(l3))
    new_list = []
    if config.use_equal:
        a = [sum(l1[idea]) / len(l1[idea]) for idea in range(len(l1))]
        for i in range(len(l2)):
            new_list.append(a[l2[i]])
    else:
        a = [[sum(l1[idx][idea]) / len(l1[idx][idea]) for idea in range(len(l1[idx]))] for idx in range(len(l1))]
        for i in range(len(l2)):  # l2 and l3 should be same length
            new_list.append(a[l3[i]-1][l2[i]])  # -1 for 0 based index
    return new_list


def check_id(new_list, id, param_size, height):
    try:
        new_list[id]
        # add additional row below
        # new_list[id] = np.vsplit([new_list[id], np.zeros(param_size)])
    except KeyError as e:  # if idea doesn't except add it to dictionary
        new_list[id] = np.zeros(param_size*height).reshape(height, param_size)
    return new_list


def dict_itr(dict, id, idx, val, add_on):
    for i in range(len(dict)):
        if add_on:
            dict[i][id][idx] += val
        else:
            dict[i][id][idx] = val


def from_3d_to_2d(arr):
    # arr.shape returns (x, y, z) --> y is the transformed length in 2d
    out_arr = arr.transpose(0, 1, 2).reshape(arr.shape[0] * arr.shape[1], -1)
    i = 0
    while i < len(out_arr):
        # check of Index Out of Bounds Exception
        try:
            out_arr[i]
        except IndexError as e:
            break
        if np.array_equal(out_arr[i], np.zeros(out_arr.shape[1])):
            out_arr = np.delete(out_arr, i, 0)
        else:
            i += 1
    return out_arr


# takes the average of the intervals
def process_bin(arr, bin_size, num_bin):
    out = np.zeros(num_bin)
    for bin in range(num_bin - 1):
        out[bin] = sum(arr[bin*bin_size:(bin+1)*bin_size]) / bin_size
    bin += 1
    out[bin] = sum(arr[bin*bin_size:]) / len(arr[bin*bin_size:])
    return out


def append_list(arr1, arr2):
    out = list(arr1)
    for i in arr2:
        out.append(i)
    return np.asarray(out)


def find_dup(arr):
    out = []
    num = []
    last = arr[0]
    last_index = 0
    for i in range(1, len(arr)):
        if last != arr[i]:
            out.append((last_index, i))
            num.append(last)
            last = arr[i]
            last_index = i
    num.append(last)
    # 2nd part to keep track of how many elements till end of arr, -1 means end
    out.append((last_index, -1))
    return np.asarray(num), np.asarray(out)


def remove_dup(arr, idx):
    out = []
    for i in idx:
        if i[1] != -1:  # if last element isn't included
            out.append(sum(arr[i[0]:i[1]])/len(arr[i[0]:i[1]]))
        else:
            out.append(sum(arr[i[0]:])/len(arr[i[0]:]))
    return np.asarray(out)
