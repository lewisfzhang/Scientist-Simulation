# optimize.py

from model import *
import random, magic
from store import *
from scipy import stats
from functions import *
import warnings as w


# scientist chooses the idea that returns the most at each step
def investing_helper(scientist, lock):
    # temp df for ideas scientist has invested in
    temp_df = pd.DataFrame(columns=['Idea Choice', 'Max Return', 'ID'])
    copy_total_start_effort(scientist, lock[0])

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
        scientist.marginal_effort = scientist.increment - scientist.curr_k  # NOTE: funding is now part of marginal!

        # Selects idea that gives the max return given equivalent "marginal" efforts
        # NOTE: See above for comments on calc_cum_returns function,
        # and exceptions on when the idea with the max return isn't chosen
        if config.switch == 3:
            # equally split scientists who need to go through funding and don't need to
            idea_choice, max_return, funding_amt = greedy_returns(scientist, scientist.must_fund, lock[1], lock[2])
        elif config.switch == 4:
            idea_choice, max_return, funding_amt = smart_returns(scientist, lock[1], lock[2])
        else:
            idea_choice, max_return, funding_amt = probabilistic_returns(scientist, scientist.must_fund, lock[1], lock[2])

        # NOTE: Commented out since this probably won't happen, just letting for loop keep on going isn't too costly
        # NOTE2: on the other hand, we need max_return to be 0 at times to encourage learning and funding parts
        # Accounts for the edge case in which max_return = 0 (implying that a
        # scientist either can't invest in ANY ideas [due to investment
        # cost barriers or an idea reaching max investment]). Effort
        # is lost and doesn't carry over to the next period
        # if max_return == 0:
        #     scientist.avail_effort = 0
        #     continue

        unpack_model_arrays_data(scientist.model, lock[0])
        idx_idea_phase = (scientist.model.idea_phase_label[idea_choice] < scientist.total_effort_start[idea_choice]).sum()
        curr_age = scientist.model.schedule.time - math.ceil(scientist.unique_id / config.N)  # same as TP - birth order in agent step function
        rel_age = int(curr_age * 2 / config.time_periods_alive)  # halflife defines young vs old
        scientist.model.total_idea_phase[idx_idea_phase][rel_age] += 1
        scientist.model.total_effort[idea_choice] += scientist.marginal_effort[idea_choice]
        scientist.model.total_perceived_returns[idea_choice] += max_return
        scientist.model.total_times_invested[idea_choice] += 1
        scientist.model.total_k[idea_choice] += scientist.curr_k[idea_choice]
        rel_age = int(scientist.current_age * 2 / config.time_periods_alive)  # halflife defines young vs old
        scientist.model.effort_invested_by_age[rel_age][idea_choice] += scientist.marginal_effort[idea_choice]
        if scientist.unique_id not in scientist.model.total_scientists_invested_helper[idea_choice]:
            scientist.model.total_scientists_invested[idea_choice] += 1
            scientist.model.total_scientists_invested_helper[idea_choice].add(scientist.unique_id)
        store_model_arrays_data(scientist.model, False, lock[0])
        scientist.marginal_invested_by_scientist[idea_choice] += scientist.marginal_effort[idea_choice]
        scientist.k_invested_by_scientist[idea_choice] += scientist.curr_k[idea_choice]
        scientist.eff_inv_in_period_marginal[idea_choice] += scientist.marginal_effort[idea_choice]
        scientist.eff_inv_in_period_k[idea_choice] += scientist.curr_k[idea_choice]
        scientist.eff_inv_in_period_funding[idea_choice] += funding_amt  # scientist.model.funding[idea_choice]
        scientist.eff_inv_in_period_f_mult[idea_choice] = scientist.funding_invested_by_scientist[idea_choice]
        scientist.avail_effort -= scientist.increment
        scientist.total_effort_start[idea_choice] += scientist.marginal_effort[idea_choice]

        # checks if idea_choice is already in the df
        if idea_choice in temp_df['Idea Choice'].values:
            row_data = temp_df.loc[temp_df['Idea Choice'] == idea_choice]
            temp_df = temp_df.drop(temp_df.index[temp_df['Idea Choice'] == idea_choice][0])
            # idea choice and ID should stay the same
            add_row = {"Idea Choice": 0, "Max Return": max_return, "ID": 0, "Marginal": scientist.marginal_effort[idea_choice]}
            row_data += add_row
            temp_df = temp_df.append(row_data, ignore_index=True)
        # if idea_choice is not in df
        else:
            row_data = {"Idea Choice": idea_choice, "Max Return": max_return, "ID": scientist.unique_id,
                        "Marginal": scientist.marginal_effort[idea_choice]}
            temp_df = temp_df.append(row_data, ignore_index=True)

        del idea_choice, max_return, no_effort_inv, scientist.curr_k, scientist.increment, row_data, idx_idea_phase, funding_amt
        scientist.increment = None
        scientist.curr_k = None

    # appending current dataframe to model investing queue
    update_investing_queue(scientist.model, temp_df, lock[3])

    # clearing data before running while loop again (no need for GC since it is run right after we exit the method)
    del scientist.total_effort_start, temp_df


def smart_returns(scientist, *lock):  # mp locks not implemented since probably not necessary
    exp_rtn, exp_val = [], []  # indexed based on each idea

    unpack_model_lists(scientist.model, lock[0])
    if config.use_equal:
        exp_val = [sum(scientist.model.exp_bayes[idea]) / len(scientist.model.exp_bayes[idea]) for idea in
                   range(len(scientist.model.exp_bayes))]
        exp_val = np.log(exp_val)
    else:
        exp_val = [sum(scientist.model.exp_bayes[scientist.unique_id - 1][idea]) / len(
            scientist.model.exp_bayes[scientist.unique_id - 1][idea]) for idea in
                   range(len(scientist.model.exp_bayes[scientist.unique_id - 1]))]
        # 0.5 shifts range from 0-1 to 0.5-1.5 so even if scientist is not oldest he does not despair
        exp_val = [0.5 + get_bayesian_formula([a, 1 - a]) for a in exp_val]
    store_model_lists(scientist.model, False, lock[0])

    # retrieving data to format into neural network acceptable parameters
    for idea in np.where(scientist.avail_ideas)[0]:
        effort = int(scientist.total_effort_start[idea])
        concav = logistic_cdf_2d(effort, idea, scientist.model.actual_returns_matrix)
        slope = get_returns(idea, scientist.perceived_returns_matrix, effort, effort+1)
        # calculates remaining funding left required
        funding_remaining = (1 - scientist.funding_invested_by_scientist[idea] /
                             (1 + scientist.model.f_mult[idea])) * scientist.model.funding[idea]
        # dict just to make things more readable at this point
        new_dict = {'avail_effort': scientist.avail_effort,
                    'learning check': int(scientist.curr_k[idea] > 0 and scientist.marginal_effort[idea] > 0),
                    'research check': int(scientist.marginal_effort[idea] > 0),
                    'funding check': 0,  # int(funding_remaining <= 0),  # False = funding done, True = funding remains
                    'scientist time alive': math.ceil(scientist.unique_id / config.N) + config.time_periods_alive - scientist.model.schedule.time,
                    'idea age': scientist.model.schedule.time - (idea // config.ideas_per_time) + 0.0001,  # to account for 0-aged ideas
                    'concav': concav,
                    'exp': exp_val[idea],
                    'slope': slope,
                    'learning k': scientist.curr_k[idea],
                    'marginal': scientist.marginal_effort[idea],
                    'funding': scientist.model.funding[idea],
                    'increment': scientist.increment,
                    'times invested': len(scientist.model.final_marginal_invested_ideas[scientist.unique_id - 1]),
                    'f_mult': scientist.funding_invested_by_scientist[idea]}
        exp_rtn.append(np.hstack(list(new_dict.values())))  # convert vstack into hstack

        # ONE WITH FUNDING, ANOTHER WITHOUT
        new_dict = {'avail_effort': scientist.avail_effort,
                    'learning check': int(scientist.curr_k[idea] > 0 and scientist.marginal_effort[idea] > 0),
                    'research check': int(scientist.marginal_effort[idea] > 0),
                    'funding check': 1,  # int(funding_remaining <= 0),  # False = funding done, True = funding remains
                    'scientist time alive': math.ceil(
                        scientist.unique_id / config.N) + config.time_periods_alive - scientist.model.schedule.time,
                    'idea age': scientist.model.schedule.time - (idea // config.ideas_per_time) + 0.0001,
                    # to account for 0-aged ideas
                    'concav': concav,
                    'exp': exp_val[idea],
                    'slope': slope,
                    'learning k': scientist.curr_k[idea],
                    'marginal': scientist.marginal_effort[idea],
                    'funding': scientist.model.funding[idea],
                    'increment': scientist.increment,
                    'times invested': len(scientist.model.final_marginal_invested_ideas[scientist.unique_id - 1]),
                    'f_mult': scientist.funding_invested_by_scientist[idea]}
        exp_rtn.append(np.hstack(list(new_dict.values())))  # convert vstack into hstack
        del effort, concav, slope, new_dict

    # exp_return simulates actual return prediction
    idea_choice, exp_return, with_funding = scientist.model.brain.process(np.asarray(exp_rtn))
    # NOTE: compute magic without this division
    exp_return /= magic.dnn_exp_slope_c

    # CHECK ON THIS, MAKE A GRAPH!!!  --> nvm, already in scatterplot residual
    # print('exp:', exp_return, 'act:', end=" ")
    del exp_rtn, exp_val

    return process_idea(scientist, with_funding, idea_choice, exp_return, lock)


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
def probabilistic_returns(scientist, with_funding, *lock):  # optimistic disregarding marginal effort available
    # Array: keeping track of all the returns of investing in each available ideas
    slope_ideas, effort_ideas, score_ideas, exp_val = [], [], [], []
    p_slope, p_effort, z_slope, z_effort, data, slopes, prob_slope, bins = \
        None, None, None, None, None, None, None, None

    # Loops over all the ideas the scientist is allowed to invest in
    # condition checks ideas where scientist.avail_ideas is TRUE
    for idea in np.where(scientist.avail_ideas)[0]:
        effort = int(scientist.total_effort_start[idea])
        effort_ideas.append(effort)
        # scientist invested one unit of effort into perceived returns matrix for idea
        slope = get_returns(idea, scientist.perceived_returns_matrix, effort, effort+1)

        # CALCULATE ACTUALLY SLOPE AS WELL!
        slope_ideas.append(slope)
        del effort, slope

    if config.switch == 0:  # percentiles
        p_slope = [stats.percentileofscore(slope_ideas, slope_ideas[i])/100 for i in range(len(slope_ideas))]
        p_effort = [stats.percentileofscore(effort_ideas, effort_ideas[i])/100 for i in range(len(effort_ideas))]

    elif config.switch == 1:  # z score
        z_slope = stats.zscore(slope_ideas)
        # catches divide by 0 error in beginning step TP 2 where all effort will be 0 --> std Dev is 0
        z_effort = [0] * len(effort_ideas) if scientist.model.schedule.time == 2 else stats.zscore(effort_ideas)

    elif config.switch == 2:  # bayesian
        unpack_model_lists(scientist.model, lock[0])
        slopes = [scientist.model.final_slope[i][scientist.unique_id - 1] for i in range(0, 2)]
        if config.use_equal:
            exp_val = [sum(scientist.model.exp_bayes[idea]) / len(scientist.model.exp_bayes[idea]) for idea in
                       range(len(scientist.model.exp_bayes))]
            exp_val = np.log(exp_val)
        else:
            exp_val = [sum(scientist.model.exp_bayes[scientist.unique_id - 1][idea]) / len(
                scientist.model.exp_bayes[scientist.unique_id - 1][idea]) for idea in
                       range(len(scientist.model.exp_bayes[scientist.unique_id - 1]))]
            # 0.5 shifts range from 0-1 to 0.5-1.5 so even if scientist is not oldest he does not despair
            exp_val = [0.5 + get_bayesian_formula([a, 1-a]) for a in exp_val]
        store_model_lists(scientist.model, False, lock[0])
        # 0 = 'm > M', 1 = 'm <= M'
        data = np.asarray([np.asarray([None, None])] * len(slope_ideas))
        prob_slope = [np.asarray([]), np.asarray([])]
        bins = [np.asarray([]), np.asarray([])]
        for i in range(0, 2):
            # scientist has never invested, so he has no data for bayesian update
            if len(slopes[i]) == 0:
                for idea in range(len(slope_ideas)):
                    data[idea][i] = 0.5
                    del idea
            else:
                # prob_slope is probability of such a slope, bins are interval
                prob_slope[i], bins[i] = np.histogram(slopes[i], bins=len(slopes[i]), density=True)
                prob_slope[i] /= sum(prob_slope[i])  # ensures max probability is 1
                # for all zero elements take average of adjacent elements
                for idx, val in enumerate(prob_slope[i]):
                    # idx here should never be 0 or last value since those intervals cover min/max
                    # temp fix for above: use try statements and treat outside as 0 (NOTE: problem is still unresolved)
                    if val == 0:
                        try:
                            left = prob_slope[i][idx-1]
                        # IndexError should not happen if program runs smoothly
                        except Exception as e:
                            left = 0
                            w.warn('check prob_slope in optimize.py')
                        try:
                            right = prob_slope[i][idx+1]
                        # IndexError should not happen if program runs smoothly
                        except Exception as e:
                            right = 0
                            w.warm('check prob_slope in optimize.py')
                        prob_slope[i][idx] = (left + right)/2
                bins[i][0] = -100000  # so least value is included in last bin
                bins[i][-1] = 100000  # so greatest value is included in last bin
                data[np.arange(len(slope_ideas)), i] = prob_slope[i][np.digitize(slope_ideas, bins[i]) - 1]

    p_score, z_score, bayes_score = 0, 0, 0
    for idea in range(len(slope_ideas)):
        # penalize low slope, high effort (high score is better idea to invest)
        if config.switch == 0:
            p_score = p_slope[idea] * (1-p_effort[idea]) * scientist.model.f_mult[idea]  # idea indices are equal?
        elif config.switch == 1:
            z_score = z_slope[idea] - z_effort[idea] * scientist.model.f_mult[idea]  # idea indices should be the same?
        elif config.switch == 2:
            power_scale = 15
            # flawed because only calculates probability you will get greater returns given current slope, not the
            # best returns? --> implemented additional multipliers to account for it
            # NOTE: still need to balance out factors so that not one is dominant
            # NOTE: add possible slides?
            bayes_score = (get_bayesian_formula(data[idea]) ** power_scale) * \
                          (slope_ideas[idea] / (exp_val[idea] ** power_scale)) * \
                          scientist.model.f_mult[idea]  # idea indices should be the same?
        score_ideas.append([p_score, z_score, bayes_score][config.switch])
    del p_score, z_score, bayes_score

    # Scalar: finds the maximum return over all the available ideas
    max_return = max(score_ideas)
    # Array: finds the index of the maximum return over all the available ideas
    idx_max_return = np.where(np.asarray(score_ideas) == max_return)[0]
    # choosing random value out of all possible values (starts at index 2+10 = 12)
    random.seed(config.seed_array[scientist.unique_id][scientist.model.schedule.time + 10])
    idea_choice = idx_max_return[random.randint(0, len(idx_max_return)-1)]

    del idx_max_return, slope_ideas, effort_ideas, z_slope, z_effort, score_ideas, p_slope, p_effort, slopes, \
        prob_slope, bins, exp_val

    return process_idea(scientist, with_funding, idea_choice, None, lock)


def process_idea(scientist, with_funding, idea_choice, exp_return, lock):
    max_return, actual_return, funding_amt = 0, 0, 0

    # even if scientist can't get returns, at least some of the remaining effort he puts in goes to learning
    if scientist.marginal_effort[idea_choice] <= 0:
        scientist.k[idea_choice] -= scientist.increment
        scientist.marginal_effort[idea_choice] = 0
        scientist.curr_k[idea_choice] = scientist.increment  # become inverse of itself for data collecting

    # calculates remaining funding left required
    funding_remaining = (1 - scientist.funding_invested_by_scientist[idea_choice]/
                         (1 + scientist.model.f_mult[idea_choice])) * scientist.model.funding[idea_choice]
    # subtract effort required for funding
    # third check ensures that learning happens before funding
    if with_funding and funding_remaining > 0 and scientist.marginal_effort[idea_choice] > 0:
        # differences between the three is essentially the proportion invested into funding
        # NOTE: below check should actually be funding left!
        if funding_remaining > scientist.marginal_effort[idea_choice]:
            if scientist.marginal_effort[idea_choice] > 0:
                # raise Exception("funding should not be greater than marginal effort")
                # print('Scientist', scientist.unique_id, 'had less marginal effort available than funding', end=" ")
                # print(scientist.model.funding[idea_choice], scientist.marginal_effort[idea_choice])

                # scientist invests all of there remaining marginal into funding for future use
                scientist.funding_invested_by_scientist[idea_choice] += \
                    (scientist.marginal_effort[idea_choice] / scientist.model.funding[idea_choice]) * \
                    scientist.model.f_mult[idea_choice] + 1 - scientist.funding_invested_by_scientist[idea_choice]
                funding_amt = scientist.marginal_effort[idea_choice]
            # implies available effort was less than original increment (see above method)
            else:
                scientist.funding_invested_by_scientist[idea_choice] += \
                    (scientist.increment / scientist.model.funding[idea_choice]) * \
                    scientist.model.f_mult[idea_choice] + 1 - scientist.funding_invested_by_scientist[idea_choice]
                funding_amt = scientist.increment
            scientist.marginal_effort[idea_choice] = 0
            # manually override k since marg_effort is 0 but above method relies on past marg_effort to determine
            # future curr_k --> by logic of this code, if scientist can reach this part, he has completed learning
            scientist.k[idea_choice] = 0
        else:
            scientist.marginal_effort[idea_choice] -= funding_remaining
            funding_amt = scientist.model.funding[idea_choice]
            # += and = should be same here since we should only be running through this once
            # NOTE: disregard above, scientist could potentially only invest partially in funding (check above if loop)
            # same as "" = scientist.model.f_mult[idea_choice] + 1 ""
            scientist.funding_invested_by_scientist[idea_choice] += scientist.model.f_mult[idea_choice] + 1 - \
                                                                    scientist.funding_invested_by_scientist[idea_choice]

    scientist.model.actual_returns_matrix = unlock_actual_returns(scientist.model, lock[1])
    # indices to calculate total return from an idea, based on marginal effort
    start_index = int(scientist.total_effort_start[idea_choice])
    stop_index = int(start_index + scientist.marginal_effort[idea_choice])
    actual_slope = get_returns(idea_choice, scientist.model.actual_returns_matrix, start_index, start_index + 1)
    concav = logistic_cdf_2d(start_index, idea_choice, scientist.model.actual_returns_matrix)

    # a scientist who can't invest fully in an idea gets 0 return (above)
    if scientist.marginal_effort[idea_choice] > 0:  # see above for earlier condition
        if exp_return is None:
            max_return = get_returns(idea_choice, scientist.perceived_returns_matrix, start_index, stop_index)
        else:
            max_return = exp_return
        actual_return = get_returns(idea_choice, scientist.model.actual_returns_matrix, start_index, stop_index)
        del start_index, stop_index

    store_actual_returns(scientist.model, lock[1])
    # multiplying by funding (scientist should only reap benefits of funding after fully investing)
    # third check ensures that learning happens before funding
    if with_funding and scientist.funding_invested_by_scientist[idea_choice] >= scientist.model.f_mult[idea_choice] + 1:
        max_return *= scientist.funding_invested_by_scientist[idea_choice]
        actual_return *= scientist.funding_invested_by_scientist[idea_choice]

    # update back to the variables/attribute of the Agent object / scientist
    unpack_model_lists(scientist.model, lock[0])
    scientist.model.final_perceived_returns_invested_ideas[scientist.unique_id-1].append(max_return)
    scientist.model.final_actual_returns_invested_ideas[scientist.unique_id-1].append(actual_return)
    scientist.model.final_slope[max_return <= actual_return][scientist.unique_id-1].append(actual_slope)
    scientist.model.final_slope[2][scientist.unique_id-1].append([max_return <= actual_return, scientist.unique_id])
    scientist.model.final_k_invested_ideas[scientist.unique_id-1].append(scientist.curr_k[idea_choice])
    scientist.model.final_marginal_invested_ideas[scientist.unique_id-1].append(scientist.marginal_effort[idea_choice])
    scientist.model.final_scientist_id[scientist.unique_id-1].append(scientist.unique_id)
    scientist.model.final_idea_idx[scientist.unique_id-1].append(idea_choice)
    scientist.model.final_concavity[scientist.unique_id-1].append(concav)
    scientist.model.final_increment[scientist.unique_id-1].append(scientist.increment)
    scientist.model.final_tp_invested[scientist.unique_id-1].append(scientist.model.schedule.time)
    store_model_lists(scientist.model, False, lock[0])

    # print(actual_return)
    del actual_return, actual_slope, concav, funding_remaining

    # returns index of the invested idea, as well as its perceived and actual returns
    # print('id:', scientist.unique_id, '\tidea:', idea_choice, '\treturn:', max_return,
    #       '\tfund:', funding_amt, '\tincrement:', scientist.increment, '\tcurr_k:', scientist.curr_k[idea_choice],
    #       '\tmarg:', scientist.marginal_effort[idea_choice], '\tf_mult:', scientist.funding_invested_by_scientist[idea_choice],
    #       '\tk:', scientist.k[idea_choice])
    return idea_choice, max_return, funding_amt


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
def greedy_returns(scientist, with_funding, *lock):  # unrealistic greedy decision process (local optimum)
    scientist.model.actual_returns_matrix = unlock_actual_returns(scientist.model, lock[1])

    # Array: keeping track of all the returns of investing in each available ideas
    final_perceived_returns_avail_ideas = []
    final_actual_returns_avail_ideas = []
    funding_amt = 0

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
        tmp_funding_mult = 1
        # this is for iterating through all ideas, so we don't want to update actual marginal effort array
        tmp_marg = scientist.marginal_effort[idea]
        if with_funding:
            if scientist.funding_invested_by_scientist[idea] < scientist.model.f_mult[idea] + 1:
                if scientist.model.funding[idea] > scientist.marginal_effort[idea]:
                    tmp_marg = 0  # scientist doesn't get effects of his funding
                else:
                    tmp_marg -= scientist.model.funding[idea]
                    tmp_funding_mult += scientist.model.f_mult[idea]
            else:
                tmp_funding_mult = scientist.funding_invested_by_scientist[idea]
        stop_index = int(start_index + tmp_marg)

        # at this point scientists have maxed out idea, no point of going further
        if stop_index > 2 * config.true_means_lam:
            final_perceived_returns_avail_ideas.append(0)
            final_actual_returns_avail_ideas.append(0)
            continue

        # calculate total return based on start and stop index
        perceived_returns = get_returns(idea, scientist.perceived_returns_matrix, start_index, stop_index)
        final_perceived_returns_avail_ideas.append(perceived_returns * tmp_funding_mult)
        actual_returns = get_returns(idea, scientist.model.actual_returns_matrix, start_index, stop_index)
        final_actual_returns_avail_ideas.append(actual_returns * tmp_funding_mult)
        del start_index, stop_index, perceived_returns, actual_returns, tmp_funding_mult, tmp_marg

    # Scalar: finds the maximum return over all the available ideas
    max_return = max(final_perceived_returns_avail_ideas)
    # Array: finds the index of the maximum return over all the available ideas
    idx_max_return = np.where(np.asarray(final_perceived_returns_avail_ideas) == max_return)[0]
    # choosing random value out of all possible values (starts at index 2+10 = 12)
    random.seed(config.seed_array[scientist.unique_id][scientist.model.schedule.time + 10])
    idea_choice = idx_max_return[random.randint(0, len(idx_max_return)-1)]

    # convert back from above
    actual_return = final_actual_returns_avail_ideas[idea_choice]

    start_index = int(scientist.total_effort_start[idea_choice])
    actual_slope = get_returns(idea_choice, scientist.model.actual_returns_matrix, start_index, start_index + 1)
    concav = logistic_cdf_2d(start_index, idea_choice, scientist.model.actual_returns_matrix)
    store_actual_returns(scientist.model, lock[1])

    # even if scientist can't get returns, at least some of the remaining effort he puts in goes to learning
    if max_return <= 0 and scientist.curr_k[idea_choice] != 0:
        scientist.k[idea_choice] = scientist.k[idea_choice] - scientist.increment if scientist.k[idea_choice] - scientist.increment > 0 else 0
        scientist.marginal_effort[idea_choice] = 0
        scientist.curr_k[idea_choice] = scientist.increment

    # calculates remaining funding left required
    funding_remaining = (1 - scientist.funding_invested_by_scientist[idea_choice] /
                         (1 + scientist.model.f_mult[idea_choice])) * scientist.model.funding[idea_choice]
    # subtract effort required for funding
    # third check ensures that learning happens before funding
    if with_funding and funding_remaining > 0 and scientist.marginal_effort[idea_choice] > 0:
        # differences between the three is essentially the proportion invested into funding
        if funding_remaining > scientist.marginal_effort[idea_choice]:
            if scientist.marginal_effort[idea_choice] > 0:
                # raise Exception("funding should not be greater than marginal effort")
                # print('Scientist', scientist.unique_id, 'had less marginal effort available than funding', end=" ")
                # print(scientist.model.funding[idea_choice], scientist.marginal_effort[idea_choice])

                # scientist invests all of there remaining marginal into funding for future use
                scientist.funding_invested_by_scientist[idea_choice] += \
                    (scientist.marginal_effort[idea_choice] / scientist.model.funding[idea_choice]) * \
                    scientist.model.f_mult[idea_choice] + 1 - scientist.funding_invested_by_scientist[idea_choice]
                funding_amt = scientist.marginal_effort[idea_choice]
            # implies available effort was less than original increment (see above method)
            else:
                scientist.funding_invested_by_scientist[idea_choice] += \
                    (scientist.increment / scientist.model.funding[idea_choice]) * \
                    scientist.model.f_mult[idea_choice] + 1 - scientist.funding_invested_by_scientist[idea_choice]
                funding_amt = scientist.increment
            scientist.marginal_effort[idea_choice] = 0
            # manually override k since marg_effort is 0 but above method relies on past marg_effort to determine
            # future curr_k --> by logic of this code, if scientist can reach this part, he has completed learning
            scientist.k[idea_choice] = 0
        else:
            scientist.marginal_effort[idea_choice] -= scientist.model.funding[idea_choice]
            funding_amt = scientist.model.funding[idea_choice]
            # += and = should be same here since we should only be running through this once
            # NOTE: disregard above, scientist could potentially only invest partially in funding (check above if loop)
            # same as "" = scientist.model.f_mult[idea_choice] + 1 ""
            scientist.funding_invested_by_scientist[idea_choice] += scientist.model.f_mult[idea_choice] + 1 - \
                                                                    scientist.funding_invested_by_scientist[idea_choice]

    # update back to the variables/attribute of the Agent object / scientist
    unpack_model_lists(scientist.model, lock[0])
    scientist.model.final_perceived_returns_invested_ideas[scientist.unique_id-1].append(max_return)
    scientist.model.final_actual_returns_invested_ideas[scientist.unique_id-1].append(actual_return)
    scientist.model.final_slope[max_return <= actual_return][scientist.unique_id-1].append(actual_slope)
    scientist.model.final_slope[2][scientist.unique_id-1].append([max_return <= actual_return, scientist.unique_id])
    scientist.model.final_k_invested_ideas[scientist.unique_id-1].append(scientist.curr_k[idea_choice])
    scientist.model.final_marginal_invested_ideas[scientist.unique_id-1].append(scientist.marginal_effort[idea_choice])
    scientist.model.final_scientist_id[scientist.unique_id-1].append(scientist.unique_id)
    scientist.model.final_idea_idx[scientist.unique_id-1].append(idea_choice)
    scientist.model.final_concavity[scientist.unique_id-1].append(concav)
    scientist.model.final_increment[scientist.unique_id-1].append(scientist.increment)
    scientist.model.final_tp_invested[scientist.unique_id-1].append(scientist.model.schedule.time)
    store_model_lists(scientist.model, False, lock[0])

    del final_perceived_returns_avail_ideas, final_actual_returns_avail_ideas, idx_max_return, funding_remaining

    # returns index of the invested idea, as well as its perceived and actual returns
    # print('id:', scientist.unique_id, '\tidea:', idea_choice, '\treturn:', max_return,
    #       '\tfund:', funding_amt, '\tincrement:', scientist.increment, '\tcurr_k:', scientist.curr_k[idea_choice],
    #       '\tmarg:', scientist.marginal_effort[idea_choice], '\tf_mult:', scientist.funding_invested_by_scientist[idea_choice],
    #       '\tk:', scientist.k[idea_choice])
    return idea_choice, max_return, funding_amt
