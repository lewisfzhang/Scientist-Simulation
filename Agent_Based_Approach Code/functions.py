# functions.py

import numpy as np

# returns the average of the total effort invested in all ideas by time period
def get_average_effort(model):
    arr = model.total_effort
    return sum(arr)/len(arr)

# returns the sum of the total effort invested in all ideas by time period
def get_total_effort(model):
    arr = model.total_effort
    return sum(arr)

# returns the average investment cost across all ideas
def get_avg_investment(model):
    arr = model.total_effort

# Input: Parameters for the logistic cumulative distribution function
# Output: Value at x of the logistic cdf defined by the location and scale parameter
def logistic_cdf(x, loc, scale):
    return 1/(1+np.exp((loc-x)/scale))

# Input:
# 1) num_ideas (scalar): number of ideas to create the return matrix for
# 2) max_of_max_inv (scalar): the maximum of all the maximum investment limits over all ideas
# 3) sds (array): the standard deviation parameters of the return curve of each idea
# 4) means (array): the mean parameters of the return curve of each idea
# Output:
# A matrix that is has dimension n x m where n = num_ideas and m = max_of_max_inv
# where each cell (i,j) contains the return based on the logistic cumulative
# distribution function of the i-th idea and the j-th extra unit of effort
def create_return_matrix(num_ideas, max_of_max_inv, sds, means):
    # Creates array of the effort units to calculate returns for
    x = np.arange(max_of_max_inv + 1)
    returns_list = []
    for i in range(num_ideas):
        # Calculates the return for an idea for all amounts of effort units
        returns = logistic_cdf(x, loc=means[i], scale=sds[i])
        # Stacks arrays horizontally
        to_subt_temp = np.hstack((0, returns[:-1]))
        # Calculates return per unit of effort
        returns = returns - to_subt_temp
        returns_list.append(returns)
    return np.array(returns_list)

# Input:
# 1) numbers (array): contains the numbers we are picking the second largest from
# Output:
# The second largest number out of the array
def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2 if count >= 2 else None