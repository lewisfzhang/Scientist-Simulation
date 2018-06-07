# optimize.py

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np
from numpy.random import poisson
from functions import *  # anything not directly tied to Mesa objects
from model import *


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

    # Scalar: limit on the amount of effort that a scientist can invest in a single idea
    # in one time period
    invest_cutoff = round(scientist.start_effort * 0.6)

    # Loops over all the ideas the scientist is allowed to invest in
    # condition checks ideas where scientist.avail_ideas is TRUE
    for idea in np.where(scientist.avail_ideas)[0]:

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
            returns = scientist.returns_matrix[idea, np.arange(start_index, stop_index)]
            total_return = sum(returns)
            final_returns_avail_ideas = np.append(final_returns_avail_ideas, total_return)

        # The case in which there are no restrictions for investing in this idea
        else:
            start_index = int(model.total_effort[idea])
            stop_index = int(start_index + scientist.marginal_effort[idea])
            returns = scientist.returns_matrix[idea, np.arange(start_index, stop_index)]
            total_return = sum(returns)
            final_returns_avail_ideas = np.append(final_returns_avail_ideas, total_return)

    # Scalar: finds the maximum return over all the available ideas
    max_return = max(final_returns_avail_ideas)

    # Array: finds the index of the maximum return over all the available ideas
    # Could be 1D array
    idx_max_return = np.where(final_returns_avail_ideas == max_return)[0]

    # Resolves edge case in which there are multiple max returns. In this case,
    # check if each idea associated with the max return has reached the
    # investment cutoff. If it has reached the cutoff, remove it from the array
    # of ideas with max returns. Then randomly choose among the remaining ideas

    # If there are ties for the maximum return
    if len(idx_max_return) > 1:

        # Array to keep track of updated index based on just the ideas that are tied for the max return
        upd_idx_max = np.array([])

        # Iterate over all indices of the array containing the indices of the tied max returns
        for idx in np.nditer(idx_max_return):

            # Deriving the correct index for the eff_inv_in_period array that contains all ideas that will
            # ever be created instead of only this scientist's available ideas that idx_max_return is in
            # reference to
            idea_choice = idx + [(model.schedule.time + 1)*model.ideas_per_time] - len(final_returns_avail_ideas)

            # If the scientist has exceeded her own limit on investing in an idea, skip to the next idea
            if scientist.eff_inv_in_period[idea_choice] >= invest_cutoff:
                continue
            # Otherwise append it to the updated-index-max array as an eligible idea
            else:
                upd_idx_max = np.append(upd_idx_max, idea_choice)

        # Randomly choose an idea over all tied, eligible ideas
        idea_choice = int(np.random.choice(upd_idx_max))

        # Return both the idea_choice (this index is relative to the eff_inv_in_period array), and the max return
        return idea_choice, max_return

    # Otherwise if there are no ties
    else:

        # Deriving the correct index for the eff_inv_in_period array that contains all ideas that will
        # ever be created instead of only this scientist's available ideas that idx_max_return is in
        # reference to
        idea_choice = idx_max_return[0] + [(model.schedule.time + 1)*model.ideas_per_time] - len(final_returns_avail_ideas)

        # Prevents scientists from investing further in a single idea if the
        # scientist has already invested at or above the investment cutoff.
        if scientist.eff_inv_in_period[idea_choice] >= invest_cutoff:
            second_max = second_largest(final_returns_avail_ideas)
            # second_max equal to 0 implies that the scientist can't invest
            # in any other ideas due to 1) ideas reaching max investment or
            # 2) ideas having too high of an investment cost given the
            # scientist's current available effort. In this edge case, the
            # scientist is allowed to continue investing in an idea past
            # the investment cutoff

            # The nested edge case where the scientist doesn't have any other ideas to invest in
            # due to other restrictions
            # These restrictions include:
            # 1) all other ideas have reached maximum investment and/or
            # 2) a scientist's marginal effort for all other ideas is <= 0,
            # implying that the scientist doesn't have enough effort to pay
            # the investment cost
            if second_max == 0:
                # Bypass the restriction on the investment cutoff in this case
                return idea_choice, max_return
            # If the scientist does have other ideas she can invest in
            else:
                # Find the idea with the second highest return
                idx_second_max = np.where(final_returns_avail_ideas == second_max)[0]

                # If there are ties for the second highest return
                if len(idx_second_max) > 1:
                    # Randomly choose between the tied ideas and derive the correct index in reference
                    # to the eff_inv_in_period array that contains all ideas that will ever be created
                    # instead of only this scientist's available ideas that idx_max_return is in reference to
                    idea_choice = int(np.random.choice(idx_second_max)) + ((model.schedule.time + 1)*model.ideas_per_time) - len(final_returns_avail_ideas)
                    return idea_choice, second_max
                # If there are no ties for the second highest return just return the only index
                else:
                    idea_choice = idx_second_max[0] + [(model.schedule.time + 1)*model.ideas_per_time] - len(final_returns_avail_ideas)
                    return idea_choice, second_max
        return idea_choice, max_return