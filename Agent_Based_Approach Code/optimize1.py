# optimize1.py

import numpy as np
from functions import *
from model import *
from random import randint


# scientist chooses the idea that returns the most at each step
def greedy_investing(scientist):
    # Scientists continue to invest in ideas until they run out of
    # available effort, or there are no more ideas to invest in
    while scientist.avail_effort > 0:
        # Array: determine which ideas scientists have invested no effort into
        no_effort_inv = (scientist.effort_invested_by_scientist == 0)

        # Array: calculate current investment costs based on prior investments;
        # has investment costs or 0 if scientist has already paid cost
        scientist.curr_k = no_effort_inv * scientist.k

        # Array (size = model.total_ideas): how much more effort can
        # be invested in a given idea based on the max investment for
        # that idea and how much all scientists have already invested
        # scientist.effort_left_in_idea = scientist.max_investment - scientist.model.total_effort

        # # Change current investment cost to 0 if a given idea has 0
        # # effort left in it. This prevents the idea from affecting
        # # the marginal effort
        # for idx, value in enumerate(scientist.effort_left_in_idea):
        #     if value == 0:
        #         scientist.curr_k[idx] = 0

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
        scientist.idea_choice, scientist.max_return, scientist.actual_return = calc_cum_returns(scientist, scientist.model)

        # Accounts for the edge case in which max_return = 0 (implying that a
        # scientist either can't invest in ANY ideas [due to investment
        # cost barriers or an idea reaching max investment]). Effort
        # is lost and doesn't carry over to the next period
        if scientist.max_return == 0:
            scientist.avail_effort = 0
            continue

        # Updates parameters after idea selection and effort expenditure
        # NOTE: self.avail_effort and self.eff_inv_in_period should be
        # updated by the increment, not by marginal effort, because the
        # increment includes investment costs. We don't care about
        # paid investment costs for the other variables
        scientist.effort_invested_by_scientist[scientist.idea_choice] += scientist.marginal_effort[scientist.idea_choice]
        scientist.eff_inv_in_period_marginal[scientist.idea_choice] += scientist.marginal_effort[scientist.idea_choice]
        scientist.eff_inv_in_period_increment[scientist.idea_choice] += scientist.increment
        scientist.avail_effort -= scientist.increment
        scientist.perceived_returns[scientist.idea_choice] += scientist.max_return  # constant in 2-period lifespan scientists
        scientist.actual_returns[scientist.idea_choice] += scientist.actual_return  # constant in 2-period lifespan scientists

        scientist.model.total_effort[scientist.idea_choice] += scientist.marginal_effort[scientist.idea_choice]
        scientist.model.effort_invested_by_age[int(scientist.current_age*2/scientist.model.time_periods_alive)][scientist.idea_choice] += scientist.marginal_effort[scientist.idea_choice]
        print("\ncurrent age", scientist.current_age, "   id", scientist.unique_id, "    step", scientist.model.schedule.time, "rel age:", int(scientist.current_age*2/scientist.model.time_periods_alive))
        print("incr", scientist.increment, "curr_k", scientist.curr_k[scientist.idea_choice], "marginal",scientist.marginal_effort[scientist.idea_choice],"\n\n\n")
        print("max return",scientist.max_return,"idea_choice",scientist.idea_choice)
        print("size", len(scientist.model.total_actual_returns))
        scientist.model.total_perceived_returns[scientist.idea_choice] += scientist.max_return
        scientist.model.total_actual_returns[scientist.idea_choice] += scientist.actual_return
        scientist.model.total_times_invested[scientist.idea_choice] += 1
        scientist.model.total_k[scientist.idea_choice] += scientist.curr_k[scientist.idea_choice]
        if scientist.curr_k[scientist.idea_choice] != 0:
            scientist.model.total_scientists_invested[scientist.idea_choice] += 1

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
    final_actual_returns_avail_ideas = []

    # Scalar: limit on the amount of effort that a scientist can invest in a single idea
    # in one time period
    # invest_cutoff = round(scientist.start_effort * input_file.prop_invest_limit)

    # Loops over all the ideas the scientist is allowed to invest in
    # condition checks ideas where scientist.avail_ideas is TRUE
    for idea in np.where(scientist.avail_ideas)[0]:
        # a scientist who can't invest fully in an idea gets 0 return
        if scientist.marginal_effort[idea] <= 0:
            final_perceived_returns_avail_ideas.append(0)
            final_actual_returns_avail_ideas.append(0)
            continue

        # indices to calculate total return from an idea, based on marginal effort
        start_index = int(model.total_effort[idea])
        stop_index = int(start_index + scientist.marginal_effort[idea])

        # at this point scientists have maxed out idea, no point of going further
        if stop_index > 2*scientist.model.true_means_lam:
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
    idx = idx_max_return[randint(0, len(idx_max_return)-1)]
    # note idx and idea_choice are the same since ((model.schedule.time + 1) * model.ideas_per_time) = len(final_perceived_returns_avail_ideas)
    idea_choice = idx + ((model.schedule.time + 1) * model.ideas_per_time) - len(final_perceived_returns_avail_ideas)
    # convert back from above
    actual_return = final_actual_returns_avail_ideas[idx]

    # update back to the variables/attribute of the Agent object / scientist
    scientist.final_perceived_returns_invested_ideas.append(max_return)
    scientist.final_actual_returns_invested_ideas.append(actual_return)
    scientist.final_k_invested_ideas.append(scientist.k[idea_choice])

    return idea_choice, max_return, actual_return
