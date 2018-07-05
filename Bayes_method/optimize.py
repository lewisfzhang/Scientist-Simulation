# optimize.py

import numpy as np
from functions import *
from model import *
import random
import config
import pandas as pd
import config
from store import *


# scientist chooses the idea that returns the most at each step
def investing_helper(scientist, lock):
    # load arrays needed
    # temp df for ideas scientist has invested in
    temp_df = pd.DataFrame(columns=['Idea Choice', 'Max Return', 'ID'])

    if config.use_multiprocessing and lock[0] is not None:  # corresponding checks if we are using multiprocessing
        lock[0].acquire()
    if config.use_store:
        scientist.total_effort_start = np.load(scientist.model.directory + 'total_effort.npy')
    else:
        scientist.total_effort_start = np.copy(scientist.model.total_effort)
    if config.use_multiprocessing and lock[0] is not None:
        lock[0].release()

    # Scientists continue to invest in ideas until they run out of
    # available effort, or there are no more ideas to invest in
    while scientist.avail_effort > 0:
        # Array: determine which ideas scientists have invested no effort into
        no_effort_inv = (scientist.marginal_invested_by_scientist == 0)

        # Array: calculate current investment costs based on prior investments;
        # has investment costs or 0 if scientist has already paid cost
        scientist.curr_k = no_effort_inv * scientist.k

        # Scalar: want to pull returns for expending self.increment units
        # of effort, where increment equals the max invest cost across
        # all ideas that haven't been invested in yet plus 1
        scientist.increment = max(scientist.curr_k[scientist.avail_ideas]) + 1

        # Edge case in which scientist doesn't have enough effort to invest in idea with greatest investment cost
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
        idea_choice, max_return = calc_cum_returns(scientist, lock[1])

        # Accounts for the edge case in which max_return = 0 (implying that a
        # scientist either can't invest in ANY ideas [due to investment
        # cost barriers or an idea reaching max investment]). Effort
        # is lost and doesn't carry over to the next period
        if max_return == 0:
            scientist.avail_effort = 0
            continue

        scientist.marginal_invested_by_scientist[idea_choice] += scientist.marginal_effort[idea_choice]
        scientist.k_invested_by_scientist[idea_choice] += scientist.curr_k[idea_choice]
        scientist.eff_inv_in_period_marginal[idea_choice] += scientist.marginal_effort[idea_choice]
        scientist.eff_inv_in_period_k[idea_choice] += scientist.curr_k[idea_choice]
        scientist.avail_effort -= scientist.increment
        scientist.total_effort_start[idea_choice] += scientist.marginal_effort[idea_choice]

        unpack_model_arrays_data(scientist.model, lock[0])

        scientist.model.total_effort[idea_choice] += scientist.marginal_effort[idea_choice]
        scientist.model.total_perceived_returns[idea_choice] += max_return
        # scientist.model.total_actual_returns[idea_choice] += actual_return
        scientist.model.total_times_invested[idea_choice] += 1
        scientist.model.total_k[idea_choice] += scientist.curr_k[idea_choice]

        rel_age = int(scientist.current_age * 2 / config.time_periods_alive)  # halflife defines young vs old
        scientist.model.effort_invested_by_age[rel_age][idea_choice] += scientist.marginal_effort[idea_choice]

        if scientist.unique_id not in scientist.model.total_scientists_invested_helper[idea_choice]:
            scientist.model.total_scientists_invested[idea_choice] += 1
            scientist.model.total_scientists_invested_helper[idea_choice].add(scientist.unique_id)

        store_model_arrays_data(scientist.model, False, lock[0])

        # checks if idea_choice is already in the df
        if idea_choice in temp_df['Idea Choice'].values:
            row_data = temp_df.loc[temp_df['Idea Choice'] == idea_choice]
            temp_df = temp_df.drop(temp_df.index[temp_df['Idea Choice'] == idea_choice][0])
            # idea choice and ID should stay the same
            add_row = {"Idea Choice": 0, "Max Return": max_return, "ID": 0}
            row_data += add_row
            temp_df = temp_df.append(row_data, ignore_index=True)
        # if idea_choice is not in df
        else:
            row_data = {"Idea Choice": idea_choice, "Max Return": max_return, "ID": scientist.unique_id}
            temp_df = temp_df.append(row_data, ignore_index=True)

        del idea_choice, max_return, no_effort_inv, scientist.curr_k, scientist.increment, row_data
        scientist.increment = None
        scientist.curr_k = None

    # appending current dataframe to model investing queue
    if config.use_multiprocessing:
        lock[2].acquire()
    investing_queue = pd.read_pickle(scientist.model.directory + 'investing_queue.pkl')
    investing_queue = investing_queue.append(temp_df, ignore_index=True)
    investing_queue.to_pickle(scientist.model.directory + 'investing_queue.pkl')
    if config.use_multiprocessing:
        lock[2].release()

    # clearing data before running while loop again (no need for GC since it is run right after we exit the method)
    del scientist.total_effort_start, temp_df, investing_queue


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
def calc_cum_returns(scientist, lock):
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
        start_index = int(scientist.total_effort_start[idea])
        stop_index = int(start_index + scientist.marginal_effort[idea])

        # at this point scientists have maxed out idea, no point of going further
        if stop_index > 2*config.true_means_lam:
            final_perceived_returns_avail_ideas.append(0)
            final_actual_returns_avail_ideas.append(0)
            continue

        # calculate total return based on start and stop index
        perceived_returns = get_returns(idea, scientist.perceived_returns_matrix, start_index, stop_index)
        final_perceived_returns_avail_ideas.append(perceived_returns)
        actual_returns = get_returns(idea, scientist.model.actual_returns_matrix, start_index, stop_index)
        final_actual_returns_avail_ideas.append(actual_returns)

    # Scalar: finds the maximum return over all the available ideas
    max_return = max(final_perceived_returns_avail_ideas)
    # Array: finds the index of the maximum return over all the available ideas
    idx_max_return = np.where(np.asarray(final_perceived_returns_avail_ideas) == max_return)[0]
    # choosing random value out of all possible values
    random.seed(config.seed_array[scientist.unique_id][scientist.model.schedule.time + 4])
    idea_choice = idx_max_return[random.randint(0, len(idx_max_return)-1)]

    # convert back from above
    actual_return = final_actual_returns_avail_ideas[idea_choice]

    if max_return == 0:
        # even if scientist can't get returns, at least some of the remaining effort he puts in goes to learning
        if scientist.curr_k[idea_choice] != 0:
            scientist.k[idea_choice] -= scientist.increment
            scientist.marginal_effort[idea_choice] = 0
            scientist.curr_k[idea_choice] = scientist.increment

    # update back to the variables/attribute of the Agent object / scientist
    scientist.model.final_perceived_returns_invested_ideas[scientist.unique_id-1].append(max_return)
    scientist.model.final_actual_returns_invested_ideas[scientist.unique_id-1].append(actual_return)
    scientist.model.final_k_invested_ideas[scientist.unique_id-1].append(scientist.curr_k[idea_choice])
    scientist.model.final_marginal_invested_ideas[scientist.unique_id-1].append(scientist.marginal_effort[idea_choice])
    scientist.model.final_scientist_id[scientist.unique_id-1].append(scientist.unique_id)
    scientist.model.final_idea_idx[scientist.unique_id-1].append(idea_choice)

    store_model_lists(scientist.model, False, lock)
    del final_perceived_returns_avail_ideas, final_actual_returns_avail_ideas, idx_max_return

    # returns index of the invested idea, as well as its perceived and actual returns
    return idea_choice, max_return
