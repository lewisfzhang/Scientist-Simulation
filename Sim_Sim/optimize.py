# optimize.py

import numpy as np
from functions import *
from model import *
from random import randint
import input_file
import pandas as pd
import input_file
from store import *


# scientist chooses the idea that returns the most at each step
def greedy_investing(scientist, lock):
    # load arrays needed
    # temp df for ideas scientist has invested in
    temp_df = pd.DataFrame(columns=['Idea Choice', 'Marginal Effort', 'Increment', 'Max Return', 'Actual Return', 'ID',
                                    'Times Invested'])

    # Scientists continue to invest in ideas until they run out of
    # available effort, or there are no more ideas to invest in
    while scientist.avail_effort > 0:
        # clearing data before running while loop again
        no_effort_inv = None
        curr_k = None
        increment = None
        idea_choice = None
        max_return = None
        actual_return = None
        row_data = None
        row = None
        gc_collect()

        # Array: determine which ideas scientists have invested no effort into
        no_effort_inv = (scientist.effort_invested_by_scientist == 0)

        # Array: calculate current investment costs based on prior investments;
        # has investment costs or 0 if scientist has already paid cost
        curr_k = no_effort_inv * scientist.k

        # Scalar: want to pull returns for expending self.increment units
        # of effort, where increment equals the max invest cost across
        # all ideas that haven't been invested in yet plus 1
        increment = max(curr_k[scientist.avail_ideas]) + 1

        # Edge case in which scientist doesn't have enough effort to invest in idea with greatest investment cost
        # POTENTIAL BIAS! LESS EFFORT SHOULD CORRESPOND TO LESS RESULTS?!
        if scientist.avail_effort < increment:
            increment = scientist.avail_effort

        # Array: contains equivalent "marginal" efforts across ideas
        # For idea with the max invest cost, marginal_effort will equal 1
        # All others - marginal_effort is equal to increment minus invest cost, if any
        scientist.marginal_effort = increment - curr_k

        # Selects idea that gives the max return given equivalent "marginal" efforts
        # NOTE: See above for comments on calc_cum_returns function,
        # and exceptions on when the idea with the max return isn't chosen
        idea_choice, max_return, actual_return = calc_cum_returns(scientist, scientist.model, lock[1])

        # Accounts for the edge case in which max_return = 0 (implying that a
        # scientist either can't invest in ANY ideas [due to investment
        # cost barriers or an idea reaching max investment]). Effort
        # is lost and doesn't carry over to the next period
        if max_return == 0:
            scientist.avail_effort = 0
            continue

        scientist.effort_invested_by_scientist[idea_choice] += scientist.marginal_effort[idea_choice]
        scientist.eff_inv_in_period_marginal[idea_choice] += scientist.marginal_effort[idea_choice]
        scientist.eff_inv_in_period_increment[idea_choice] += increment
        scientist.avail_effort -= increment

        unpack_model_arrays_data(scientist.model, lock[0])
        scientist.model.total_effort[idea_choice] += scientist.marginal_effort[idea_choice]
        scientist.model.effort_invested_by_age[int(scientist.current_age * 2 / input_file.time_periods_alive)] \
            [idea_choice] += scientist.marginal_effort[idea_choice]  # halflife defines young vs old
        scientist.model.total_perceived_returns[idea_choice] += max_return
        scientist.model.total_actual_returns[idea_choice] += actual_return
        scientist.model.total_times_invested[idea_choice] += 1
        scientist.model.total_k[idea_choice] += curr_k[idea_choice]
        store_model_arrays_data(scientist.model, False, lock[0])

        # checks if idea_choice is already in the df
        if idea_choice in temp_df['Idea Choice'].values:
            row_data = temp_df.loc[temp_df['Idea Choice'] == idea_choice]
            temp_df = temp_df.drop(temp_df.index[temp_df['Idea Choice'] == idea_choice][0])
            add_row = {"Idea Choice": 0, "Marginal Effort": scientist.marginal_effort[idea_choice],
                        "Increment": increment, "Max Return": max_return, "Actual Return": actual_return,
                        "ID": 0, "Times Invested": 1}  # idea choice and ID should stay the same
            row_data += add_row
            temp_df = temp_df.append(row_data, ignore_index=True)
        # if idea_choice is not in df
        else:
            row_data = {"Idea Choice": idea_choice, "Marginal Effort": scientist.marginal_effort[idea_choice],
                        "Increment": increment, "Max Return": max_return, "Actual Return": actual_return,
                        "ID": scientist.unique_id, "Times Invested": 1}
            temp_df = temp_df.append(row_data, ignore_index=True)

    unpack_model_arrays_data(scientist.model, lock[0])
    scientist.model.total_scientists_invested[idea_choice] += 1
    store_model_arrays_data(scientist.model, False, lock[0])

    # appending current dataframe to model investing queue
    if input_file.use_multiprocessing:
        lock[2].acquire()
    investing_queue = pd.read_pickle('tmp/model/investing_queue.pkl')
    investing_queue = investing_queue.append(temp_df, ignore_index=True)
    investing_queue.to_pickle('tmp/model/investing_queue.pkl')
    if input_file.use_multiprocessing:
        lock[2].release()

    # clearing data before running while loop again (no need for GC since it is run right after we exit the method)
    del no_effort_inv, curr_k, increment, idea_choice, max_return, actual_return, row_data, row, temp_df, investing_queue


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
def calc_cum_returns(scientist, model, lock):
    unpack_model_lists(scientist.model, lock)
    # Array: keeping track of all the returns of investing in each available ideas
    final_perceived_returns_avail_ideas = []
    final_actual_returns_avail_ideas = []

    # Loops over all the ideas the scientist is allowed to invest in
    # condition checks ideas where scientist.avail_ideas is TRUE
    for idea in np.where(scientist.avail_ideas)[0]:
        # a scientist who can't invest fully in an idea gets 0 return
        if scientist.marginal_effort[idea] <= 0:
            final_perceived_returns_avail_ideas.append(0)
            final_actual_returns_avail_ideas.append(0)
            continue

        # indices to calculate total return from an idea, based on marginal effort
        start_index = int(scientist.effort_invested_by_scientist[idea])
        stop_index = int(start_index + scientist.marginal_effort[idea])

        # at this point scientists have maxed out idea, no point of going further
        if stop_index > 2*input_file.true_means_lam:
            final_perceived_returns_avail_ideas.append(0)
            final_actual_returns_avail_ideas.append(0)
            continue

        # calculate total return based on start and stop index
        perceived_returns = scientist.perceived_returns_matrix[idea, np.arange(start_index, stop_index)]
        final_perceived_returns_avail_ideas.append(sum(perceived_returns))
        actual_returns = model.actual_returns_matrix[idea, np.arange(start_index, stop_index)]
        final_actual_returns_avail_ideas.append(sum(actual_returns))

    # Scalar: finds the maximum return over all the available ideas
    max_return = max(final_perceived_returns_avail_ideas)
    # Array: finds the index of the maximum return over all the available ideas
    idx_max_return = np.where(np.asarray(final_perceived_returns_avail_ideas) == max_return)[0]
    # choosing random value out of all possible values
    idea_choice = idx_max_return[randint(0, len(idx_max_return)-1)]

    # convert back from above
    actual_return = final_actual_returns_avail_ideas[idea_choice]

    # update back to the variables/attribute of the Agent object / scientist
    scientist.model.final_perceived_returns_invested_ideas[scientist.unique_id-1].append(max_return)
    scientist.model.final_actual_returns_invested_ideas.append(actual_return)
    scientist.model.final_k_invested_ideas.append(scientist.k[idea_choice])

    store_model_lists(scientist.model, False, lock)
    final_perceived_returns_avail_ideas = None
    final_actual_returns_avail_ideas = None
    idx_max_return = None

    # returns index of the invested idea, as well as its perceived and actual returns
    return idea_choice, max_return, actual_return
