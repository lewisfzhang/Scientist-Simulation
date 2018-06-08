# optimize.py

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np
from numpy.random import poisson
from functions import *  # anything not directly tied to Mesa objects
from model import *
import input_file


# Function to calculate "marginal" returns for available ideas, taking into account investment costs
# Input:
# 1) avail_ideas (array): indexes which ideas are available to a given scientist
# 2) total_effort (array):  contains cumulative effort already expended by all scientists
# 3) max_investment (array): contains the max possible investment for each idea
# 4) marginal_effort (array): "equivalent" marginal efforts, which equals
#    the max, current investment cost plus one minus individual investment costs for available ideas
#
# Output:
# 1) idx_max_return (scalar): the index of the idea the scientist chose to invest in
# 2) max_return (scalar): the perceived return of the associated, chosen idea
def calc_cum_returns(scientist, model):
    # Array: keeping track of all the returns of investing in each available ideas
    final_returns_avail_ideas = np.array([])
    final_k_avail_ideas = np.array([])
    # Scalar: limit on the amount of effort that a scientist can invest in a single idea
    # in one time period
    invest_cutoff = round(scientist.start_effort * input_file.prop_invest_limit)

    # Loops over all the ideas the scientist is allowed to invest in
    # condition checks ideas where scientist.avail_ideas is TRUE
    for idea in np.where(scientist.avail_ideas)[0]:

        final_k_avail_ideas = np.append(final_k_avail_ideas, scientist.k[idea])
        # OR Conditions
        # 1st) Edge case in which scientist doesn't have enough effort to invest in
        # an idea given the investment cost
        # 2nd) Ensures that effort invested in ideas doesn't go over max investment limit
        if scientist.marginal_effort[idea] <= 0 or scientist.effort_left_in_idea[idea] == 0:
            final_returns_avail_ideas = np.append(final_returns_avail_ideas, 0)

        # For instances in which a scientist's marginal effort exceeds the
        # effort left in a given idea, calculate the returns for investing
        # exactly the effort left
        elif scientist.marginal_effort[idea] > scientist.effort_left_in_idea[idea]:
            start_index = int(model.total_effort[idea])
            stop_index = int(start_index + scientist.effort_left_in_idea[idea])
            returns = scientist.perceived_returns_matrix[idea, np.arange(start_index, stop_index)]
            total_return = sum(returns)
            final_returns_avail_ideas = np.append(final_returns_avail_ideas, total_return)

        # The case in which there are no restrictions for investing in this idea
        else:
            start_index = int(model.total_effort[idea])
            stop_index = int(start_index + scientist.marginal_effort[idea])
            returns = scientist.perceived_returns_matrix[idea, np.arange(start_index, stop_index)]
            total_return = sum(returns)
            final_returns_avail_ideas = np.append(final_returns_avail_ideas, total_return)

    # that way, I can delete elements in the copy
    # scale because returns are so small (range of distribution curves is from 0-1
    final_returns_avail_ideas = 1000*final_returns_avail_ideas
    final_returns_avail_ideas_copy = np.copy(final_returns_avail_ideas)

    # update back
    scientist.final_returns_avail_ideas = final_returns_avail_ideas
    scientist.final_k_avail_ideas = final_k_avail_ideas

    while True:
        # Scalar: finds the maximum return over all the available ideas
        max_return = max(final_returns_avail_ideas_copy)

        # Array: finds the index of the maximum return over all the available ideas
        # Could be 1D array
        idx_max_return = np.where(final_returns_avail_ideas_copy == max_return)[0]

        # Array to keep track of updated index based on just the ideas that are tied for the max return
        upd_idx_max = np.array([])

        # Iterate over all indices of the array containing the indices of the tied max returns
        for idx in np.nditer(idx_max_return):
            # Deriving the correct index for the eff_inv_in_period array that contains all ideas that will
            # ever be created instead of only this scientist's available ideas that idx_max_return is in
            # reference to
            idea_choice = idx + [(model.schedule.time + 1) * model.ideas_per_time] - len(final_returns_avail_ideas)

            # If the scientist has exceeded her own limit on investing in an idea, skip to the next idea
            # Otherwise append it to the updated-index-max array as an eligible idea
            if scientist.eff_inv_in_period[idea_choice] < invest_cutoff:
                upd_idx_max = np.append(upd_idx_max, idea_choice)

        if np.size(upd_idx_max) > 0:
            idea_choice = int(np.random.choice(upd_idx_max))
            break;
        else:
            # deletes all the max values in this cycle
            final_returns_avail_ideas_copy = np.delete(final_returns_avail_ideas_copy, idx_max_return)

            # SAFETY: ensures while loop doesn't run forever
            if np.size(final_returns_avail_ideas_copy) == 0:
                break;

    return idea_choice, max_return
