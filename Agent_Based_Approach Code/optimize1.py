# optimize1.py

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np
from numpy.random import poisson
from functions import *  # anything not directly tied to Mesa objects
from model import *
import input_file
import random

# scientist chooses the idea that returns the most at each step
def greedy_investing(scientist):
    # Scientists continue to invest in ideas until they run out of
    # available effort, or there are no more ideas to invest in
    while scientist.avail_effort > 0:
        # print('BRUH IM HERE', page_counter())
        # Array: determine which ideas scientists have invested no effort into
        no_effort_inv = (scientist.effort_invested_by_scientist == 0)

        # Array: calculate current investment costs based on prior investments;
        # has investment costs or 0 if scientist has already paid cost
        scientist.curr_k = no_effort_inv * scientist.k

        # Array (size = model.total_ideas): how much more effort can
        # be invested in a given idea based on the max investment for
        # that idea and how much all scientists have already invested
        scientist.effort_left_in_idea = scientist.max_investment - scientist.model.total_effort

        # Change current investment cost to 0 if a given idea has 0
        # effort left in it. This prevents the idea from affecting
        # the marginal effort
        for idx, value in enumerate(scientist.effort_left_in_idea):
            if value == 0:
                scientist.curr_k[idx] = 0

        # Scalar: want to pull returns for expending self.increment units
        # of effort, where increment equals the max invest cost across
        # all ideas that haven't been invested in yet plus 1
        scientist.increment = max(scientist.curr_k[scientist.avail_ideas]) + 1

        # Edge case in which scientist doesn't have enough effort to invest
        # in idea with greatest investment cost
        # POTENTIAL BIAS! LESS EFFORT SHOULD CORRESPOND TO LESS RESULTS?!
        if scientist.avail_effort < scientist.increment:
            scientist.increment = scientist.avail_effort

        # Array: contains equivalent "marginal" efforts across ideas
        # For idea with the max invest cost, marginal_effort will equal 1
        # All others - marginal_effort is equal to increment minus invest cost, if any
        scientist.marginal_effort = scientist.increment - scientist.curr_k

        # Selects idea that gives the max return given equivalent "marginal" efforts
        # NOTE: See above for comments on calc_cum_returns function,
        # and exceptions on when the idea with the max return isn't chosen
        scientist.idea_choice, scientist.max_return = calc_cum_returns(scientist, scientist.model)

        # Accounts for the edge case in which max_return = 0 (implying that a
        # scientist either can't invest in ANY ideas [due to investment
        # cost barriers or an idea reaching max investment]). Effort
        # is lost and doesn't carry over to the next period
        if scientist.max_return == 0:
            scientist.avail_effort = 0
            continue

        # Accounts for the edge case in which scientist chooses to invest in
        # an idea that has less effort remaining than a scientist's
        # marginal effort
        if scientist.marginal_effort[scientist.idea_choice] > scientist.effort_left_in_idea[scientist.idea_choice]:
            scientist.marginal_effort[scientist.idea_choice] = scientist.effort_left_in_idea[scientist.idea_choice]
            scientist.increment = scientist.curr_k[scientist.idea_choice] + scientist.marginal_effort[scientist.idea_choice]

        # Updates parameters after idea selection and effort expenditure
        # NOTE: self.avail_effort and self.eff_inv_in_period should be
        # updated by the increment, not by marginal effort, because the
        # increment includes investment costs. We don't care about
    # paid investment costs for the other variables
        scientist.model.total_effort[scientist.idea_choice] += scientist.marginal_effort[scientist.idea_choice]
        scientist.effort_invested_by_scientist[scientist.idea_choice] += scientist.marginal_effort[scientist.idea_choice]
        scientist.model.effort_invested_by_age[scientist.current_age][scientist.idea_choice] += scientist.marginal_effort[scientist.idea_choice]
        scientist.eff_inv_in_period_marginal[scientist.idea_choice] += scientist.marginal_effort[scientist.idea_choice]
        scientist.eff_inv_in_period_increment[scientist.idea_choice] += scientist.increment
        scientist.avail_effort -= scientist.increment
        scientist.perceived_returns[scientist.idea_choice] += scientist.max_return  # constant in 2-period lifespan scientists

    # system debugging print statements
    # print("\ncurrent age", scientist.current_age, "   id", scientist.unique_id, "    step", scientist.model.schedule.time,
    #       "\navail ideas array:",scientist.avail_ideas.tolist(),
    #       "\ntotal effort invested array:", scientist.effort_invested_by_scientist.tolist(),
    #       "\neffort invested in time period (increment):", scientist.eff_inv_in_period_increment.tolist(),
    #       "\neffort invested in time period (marginal):", scientist.eff_inv_in_period_marginal.tolist(),
    #       "\ncurrent cost across ideas:", scientist.curr_k)

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
    final_perceived_returns_avail_ideas = []
    final_k_avail_ideas = []
    final_actual_returns_avail_ideas = []

    # Scalar: limit on the amount of effort that a scientist can invest in a single idea
    # in one time period
    invest_cutoff = round(scientist.start_effort * input_file.prop_invest_limit)

    # Loops over all the ideas the scientist is allowed to invest in
    # condition checks ideas where scientist.avail_ideas is TRUE
    for idea in np.where(scientist.avail_ideas)[0]:
        final_k_avail_ideas.append(scientist.k[idea])
        # OR Conditions
        # 1st) Edge case in which scientist doesn't have enough effort to invest in
        # an idea given the investment cost
        # 2nd) Ensures that effort invested in ideas doesn't go over max investment limit
        if scientist.marginal_effort[idea] <= 0 or scientist.effort_left_in_idea[idea] == 0:
            final_perceived_returns_avail_ideas.append(0)
            final_actual_returns_avail_ideas.append(0)

        else:
            start_index = int(model.total_effort[idea])

            # For instances in which a scientist's marginal effort exceeds the
            # effort left in a given idea, calculate the returns for investing
            # exactly the effort left
            if scientist.marginal_effort[idea] > scientist.effort_left_in_idea[idea]:
                stop_index = int(start_index + scientist.effort_left_in_idea[idea])

            # The case in which there are no restrictions for investing in this idea
            else:
                stop_index = int(start_index + scientist.marginal_effort[idea])

            perceived_returns = scientist.perceived_returns_matrix[idea, np.arange(start_index, stop_index)]
            final_perceived_returns_avail_ideas.append(sum(perceived_returns))
            actual_returns = model.actual_returns_matrix[idea, np.arange(start_index, stop_index)]
            final_actual_returns_avail_ideas.append(sum(actual_returns))

    # scale because returns are so small (range of distribution curves is from 0-1)
    final_perceived_returns_avail_ideas = [100*i for i in final_perceived_returns_avail_ideas]
    final_actual_returns_avail_ideas = [100*i for i in final_actual_returns_avail_ideas]

    # that way, I can delete elements in the copy
    final_perceived_returns_avail_ideas_copy = np.asarray(final_perceived_returns_avail_ideas)

    # update back
    scientist.final_perceived_returns_avail_ideas = append_list(scientist.final_perceived_returns_avail_ideas, final_perceived_returns_avail_ideas)
    scientist.final_k_avail_ideas = append_list(scientist.final_k_avail_ideas, final_k_avail_ideas)
    scientist.final_actual_returns_avail_ideas = append_list(scientist.final_actual_returns_avail_ideas, final_actual_returns_avail_ideas)

    while True:
        # Scalar: finds the maximum return over all the available ideas
        max_return = max(final_perceived_returns_avail_ideas_copy)

        # Array: finds the index of the maximum return over all the available ideas
        # Could be 1D array
        idx_max_return = np.where(final_perceived_returns_avail_ideas_copy == max_return)[0]

        # Array to keep track of updated index based on just the ideas that are tied for the max return
        upd_idx_max = []

        # Iterate over all indices of the array containing the indices of the tied max returns
        for idx in np.nditer(idx_max_return):
            # Deriving the correct index for the eff_inv_in_period array that contains all ideas that will
            # ever be created instead of only this scientist's available ideas that idx_max_return is in
            # reference to
            idea_choice = idx + [(model.schedule.time + 1) * model.ideas_per_time] - len(final_perceived_returns_avail_ideas)

            # If the scientist has exceeded her own limit on investing in an idea, skip to the next idea
            # Otherwise append it to the updated-index-max array as an eligible idea
            if scientist.eff_inv_in_period_increment[idea_choice] < invest_cutoff:
                upd_idx_max.append(idea_choice)

        if len(upd_idx_max) > 0:
            idea_choice = random.choice(upd_idx_max)
            break

        else:
            # deletes all the max values in this cycle
            final_perceived_returns_avail_ideas_copy = np.delete(final_perceived_returns_avail_ideas_copy, idx_max_return)

            # SAFETY: ensures while loop doesn't run forever
            if np.size(final_perceived_returns_avail_ideas_copy) == 0:
                break

    return idea_choice, max_return
