# final.py

from functions import *
from store import *
import warnings as w


def main():
    model = load_model()
    collect_vars(model, True)


# for data collecting after model has finished running
def collect_vars(model, store_big_data):
    f_print("\n\ndone with step", model.schedule.time - 1, '... now running collect_vars')
    start = timeit.default_timer()

    if config.use_store_model is True:
        agent_vars = pd.read_pickle(model.directory + 'agent_vars_df.pkl')
    else:
        agent_vars = model.agent_df
        model.agent_df.to_pickle(model.directory + 'agent_vars_df.pkl')
        model.model_df.to_pickle(model.directory + 'model_vars_df.pkl')
        np.save(model.directory + 'effort_invested_by_age.npy', model.effort_invested_by_age)

    # <editor-fold desc="Part 1: ideas">
    unpack_model_arrays_data(model, None)
    unlock_actual_returns(model, None)
    idea = range(0, model.total_ideas, 1)
    tp = np.arange(model.total_ideas) // config.ideas_per_time
    prop_invested = rounded_tuple(model.total_effort / inv_logistic_cdf(0.99, model.actual_returns_matrix[2], model.actual_returns_matrix[1]))
    avg_k = np.round(divide_0(model.total_k, model.total_scientists_invested), 2)
    total_perceived_returns = np.round(model.total_perceived_returns, 2)
    total_actual_returns = np.round(model.total_actual_returns, 2)
    idea_phase = divide_0(model.total_idea_phase, sum(model.total_idea_phase))
    ideas_dict = {"idea": idea,
                  "TP": tp,
                  "scientists_invested": model.total_scientists_invested,
                  "times_invested": model.total_times_invested,
                  "avg_k": avg_k,
                  "total_effort (marginal)": rounded_tuple(model.total_effort),
                  "prop_invested": prop_invested,
                  "total_pr": total_perceived_returns,
                  "total_ar": total_actual_returns}
    pd.DataFrame.from_dict(ideas_dict).replace(np.nan, '', regex=True).to_pickle(model.directory + 'ideas.pkl')
    prop_invested = np.asarray(list(filter(lambda a: a != 0, prop_invested)))
    np.save(model.directory + 'prop_invested.npy', prop_invested)
    np.save(model.directory + 'prop_remaining.npy', 1 - prop_invested)
    np.save(model.directory + 'idea_phase.npy', idea_phase)
    store_model_arrays_data(model, False, None)
    store_actual_returns(model, None)
    del ideas_dict, idea, tp, prop_invested, avg_k, total_perceived_returns, total_actual_returns, idea_phase
    # </editor-fold>

    # <editor-fold desc="Part 2: ind_ideas">
    unpack_model_lists(model, None)
    ind_ideas_dict = {"idea_idx": rounded_tuple(flatten_list(model.final_idea_idx)),
                      "scientist_id": rounded_tuple(flatten_list(model.final_scientist_id)),
                      "agent_k_invested_ideas": rounded_tuple(flatten_list(model.final_k_invested_ideas)),
                      "agent_marginal_invested_ideas": rounded_tuple(flatten_list(model.final_marginal_invested_ideas)),
                      "agent_perceived_return_invested_ideas": rounded_tuple(
                          flatten_list(model.final_perceived_returns_invested_ideas)),
                      "agent_actual_return_invested_ideas": rounded_tuple(
                          flatten_list(model.final_actual_returns_invested_ideas)),
                      "slopes": rounded_tuple(flatten_list(flat_2d(model.final_slope))),
                      "exp": rounded_tuple(expand_2d(model.exp_bayes, model.final_idea_idx, model.final_scientist_id)),
                      "increment": rounded_tuple(flatten_list(model.final_increment)),
                      # not sure if multiplying concav by 10^6 will affect the neural net / big data
                      "concav": rounded_tuple(np.asarray(flatten_list(model.final_concavity)) * 10**6),
                      "tp": rounded_tuple(flatten_list(model.final_tp_invested))}
    pd.DataFrame.from_dict(ind_ideas_dict).to_pickle(model.directory + 'ind_ideas.pkl')
    # df = df.reset_index().sort_values(['scientist_id', 'idea_idx']).set_index('index').\
    #     rename_axis(None).reset_index(drop=True)
    if config.use_store_model is False:
        with open(model.directory + "final_perceived_returns_invested_ideas.txt", "wb") as fp:
            pickle.dump(model.final_perceived_returns_invested_ideas, fp)
    store_model_lists(model, False, None)
    del ind_ideas_dict
    # </editor-fold>

    if store_big_data:
        df = agent_vars.swaplevel('AgentID', 'Step').dropna().replace('\r\n', '', regex=True).sort_index(axis=0, sort_remaining=True)
        df2 = pd.read_pickle(model.directory + 'ind_ideas.pkl')
        df.to_html('tmp/agent.html')
        print('running big data collection...')
        to_big_data(df, df2)
        print('finished big data collection...')

    # <editor-fold desc="Part 3: social_output, ideas_entered">
    ind_vars = pd.read_pickle(model.directory + 'ind_ideas.pkl')
    actual_returns = agent_vars[agent_vars['Actual Returns'].str.startswith("{", na=False)]['Actual Returns']

    returns_tracker = np.zeros(model.num_scientists)
    # getting total returns from each scientists in their entire lifetime
    # idx format: (step, agent id), val is a dictionary is in string format
    for idx, val in actual_returns.items():
        agent_id = idx[1]
        last_bracket = 0
        for i in range(val.count('idea')):
            left_bracket = val[last_bracket:].index('{')
            right_bracket = val[last_bracket:].index('}') + 1
            returns = str_to_dict(val[last_bracket:][left_bracket:right_bracket])['returns']
            last_bracket += right_bracket
            returns_tracker[agent_id - 1] += returns
            del returns, left_bracket, right_bracket, i
        del agent_id, last_bracket, idx, val

    curr_id = 1
    counter_x = 0
    x_var = [0]
    y_var = [0]
    for idx, val in enumerate(ind_vars['scientist_id']):
        if curr_id != val:
            curr_id = val
            while len(x_var) <= counter_x:
                x_var.append(0)
                y_var.append(0)
            x_var[counter_x] += 1
            y_var[counter_x] += returns_tracker[curr_id - 1]
            counter_x = 0
        if ind_vars['agent_k_invested_ideas'][idx] != 0:
            counter_x += 1
        del idx, val
    # catch last scientist when for loop exits
    while len(x_var) <= counter_x:
        x_var.append(0)
        y_var.append(0)
    x_var[counter_x] += 1
    y_var[counter_x] += returns_tracker[curr_id - 1]
    counter_x = 0
    # save to model directory
    np.save(model.directory + 'social_output.npy', np.asarray(y_var))
    np.save(model.directory + 'ideas_entered.npy', np.asarray(x_var))
    print("TOTAL ACTUAL RETURN:", sum(y_var), sum(returns_tracker))
    del ind_vars, actual_returns, returns_tracker, curr_id, counter_x, x_var, y_var
    # </editor-fold>

    # <editor-fold desc="Part 4: prop_age">
    agent_vars = agent_vars.replace(np.nan, '', regex=True)
    # format: [prop paying k][total num of scientists] || two rows, TP_alive columns
    age_tracker = np.zeros(2 * config.time_periods_alive).reshape(2, config.time_periods_alive)
    for idx, val in agent_vars['Effort Invested In Period (K)'].items():
        curr_age = idx[0] - math.ceil(idx[1] / config.N)  # same as TP - birth order in agent step function

        # if statements should only pass if curr_age is within range in the array
        if val != '':
            # total number of ideas / occurrences
            num_ideas = agent_vars.loc[idx[0]].loc[idx[1]]['Effort Invested In Period (Marginal)'].count('idea')
            age_tracker[1][curr_age] += num_ideas  # DEPENDS ON WHAT JAY WANTS --> (could use 1)
            del num_ideas
            # checks those that paid k
            if val[0] == '{':
                age_tracker[0][curr_age] += val.count('idea')
        del idx, val, curr_age
    prop_age = divide_0(age_tracker[0], age_tracker[1])
    np.save(model.directory + 'prop_age.npy', prop_age)
    del prop_age, age_tracker
    # </editor-fold>

    # <editor-fold desc="Part 5: marginal_effort_by_age, prop_idea">
    unpack_model_arrays(model, None)
    agent_marg = agent_vars[agent_vars['Effort Invested In Period (Marginal)'].
        str.startswith("{", na=False)]['Effort Invested In Period (Marginal)']

    marginal_effort = [[0, 0]]  # format: [young, old]
    prop_idea = [0]
    total_ideas = 0
    for idx, val in agent_marg.items():
        last_bracket = 0
        for i in range(val.count('idea')):
            left_bracket = val[last_bracket:].index('{')
            right_bracket = val[last_bracket:].index('}') + 1
            effort = str_to_dict(val[last_bracket:][left_bracket:right_bracket])['effort']
            idea = str_to_dict(val[last_bracket:][left_bracket:right_bracket])['idea']
            last_bracket += right_bracket

            idea_age = idx[0] - model.idea_periods[idea]  # current tp - tp born
            curr_age = idx[0] - math.ceil(idx[1] / config.N)  # same as model TP - scientist birth order
            rel_age = int(curr_age * 2 / config.time_periods_alive)  # halflife defines young vs old

            while len(marginal_effort) <= idea_age:
                marginal_effort.append([0, 0])
                prop_idea.append(0)

            marginal_effort[idea_age][rel_age] += effort
            prop_idea[idea_age] += 1
            total_ideas += 1
            del left_bracket, right_bracket, effort, idea, idea_age, curr_age, rel_age
        del idx, val, last_bracket

    prop_idea = np.asarray(prop_idea) / total_ideas
    marginal_effort = flatten_list(marginal_effort)
    marginal_effort = np.asarray([marginal_effort[::2], marginal_effort[1::2]])
    np.save(model.directory + "marginal_effort_by_age.npy", marginal_effort)
    np.save(model.directory + "prop_idea.npy", prop_idea)
    store_model_arrays(model, False, None)
    del agent_vars, agent_marg, marginal_effort, prop_idea, total_ideas
    # </editor-fold>

    f_print("time elapsed:", timeit.default_timer() - start)


def to_big_data(df, df2):  # df2 = ind_ideas
    has_past = False  # NOTE: IMPORTANT SWITCH FOR AI!!!
    last_scientist = 1  # first scientists

    # format (life):
    #   0. effort_amount (all 3 combined),  --> essentially avail effort, increment
    #   1. learning (k)  [0,1]
    #   2. research (marginal)  [0,1]
    #   3. funding  [0,1]
    #   4. idea_age,
    #   5. idea_phase (based on concavity)
    #   6. exp_num_scientists/scientist_age_bayes_prop (depending on the switch),
    #   7. ***time till death (simulates TP)***,
    #   8. slope
    #   9. k,
    #   10. marginal,
    #   11. funding_k,
    #   12. avg increment
    #   13. times invested
    #   14. actual_return (the output)
    param_size = 15
    current_data = None
    new_list = {}
    tp_1 = 2  # first active tp for scientist
    # keeps track of number of tp that a scientist is active
    tp_active = len(df.loc[1].index)  # Agent 1 should be the first agent in the df
    for idx, val in df.iterrows():
        # print(idx)
        if idx[0] != last_scientist:
            last_scientist = idx[0]
            tp_1 = idx[1]
            tp_active = len(df.loc[idx[0]].index)
            if new_list != {}:  # check that row was not empty
                # for nl in list(new_list.values()):  # a list of np arrays
                append_data = np.asarray(list(new_list.values()))  # 3D array
                if current_data is None:
                    current_data = from_3d_to_2d(append_data)
                else:
                    # print(current_data.shape, append_data.shape)
                    current_data = np.concatenate((current_data, from_3d_to_2d(append_data)))
                new_list = {}
        np_idx = idx[1] - tp_1
        id_set = set()  # keeps track of all ideas
        # NOTE: all +='s are the same as ='s now since we want TP
        # NOTE: don't change order of for loops! (order matters here)
        for new_dict in process_dict(val['Effort Invested In Period (Marginal)']):
            id = str(new_dict['idea'])
            id_set.add(id)
            new_list = check_id(new_list, id, param_size, tp_active)
            new_list[id][np_idx][0] += new_dict['effort']
            new_list[id][np_idx][10] += new_dict['effort']
            new_list[id][np_idx][2] = 1  # True
        for new_dict in process_dict(val['Effort Invested In Period (K)']):
            id = str(new_dict['idea'])
            id_set.add(id)
            new_list = check_id(new_list, id, param_size, tp_active)
            # don't need to modify 0's which are by default False
            # above concern is cancelled out since idea only needs to
            # be updated once when it is first invested before learning
            new_list[id][np_idx][1] = 1
            # confirm with jay that we want increment (marginal + k + funding)
            new_list[id][np_idx][0] += new_dict['effort']
            new_list[id][np_idx][9] += new_dict['effort']
        for new_dict in process_dict(val['Effort Invested In Period (Funding)']):
            id = str(new_dict['idea'])
            id_set.add(id)
            new_list = check_id(new_list, id, param_size, tp_active)
            # don't need to modify 0's which are by default False
            # above concern is cancelled out since idea only needs to
            # be updated once when it is first invested before learning
            new_list[id][np_idx][3] = 1
            # confirm with jay that we want increment (marginal + k + funding)
            new_list[id][np_idx][0] += new_dict['effort']
            new_list[id][np_idx][11] += new_dict['effort']
        for id in id_set:
            if new_list[id][np_idx][4] == 0:  # we count idea age based on the first time a scientists invests in the idea
                new_list[id][np_idx][4] = idx[1] - (int(id) // config.ideas_per_time) + 0.0001  # to account for 0-aged ideas
            if new_list[id][np_idx][7] == 0:  # scientist can't be dead in 0 years or else he would not be here!
                # should be positive
                new_list[id][np_idx][7] = math.ceil(idx[0] / config.N) + config.time_periods_alive - idx[1]  # curr TP = idx[1]
                if math.ceil(idx[0] / config.N) + config.time_periods_alive - idx[1] < 0:
                    w.warn("scientist time left is negative! see pack_data.py")
            # use idea_idx to locate certain values faster for indeces 5 and 6 of new_list dictionary
            row = df2.loc[(df2['scientist_id'] == idx[0]) & (df2['idea_idx'] == new_dict['idea'])]
            mess = '{} value should not be 0! (see entry '+str(row.index[0])+')'
            if new_list[id][np_idx][5] == 0:
                # positive concav means early stage, and vice versa
                # new_list[id][5] = 0.1 if row.loc[row.index[0]]['concav'] > 0 else new_list[id][5] = 0.9
                # method 2: try inverse of concavity
                new_list[id][np_idx][5] = -row.loc[row.index[0]]['concav']
                if -row.loc[row.index[0]]['concav'] == 0:
                    w.warn(mess.format('concav'))
            if new_list[id][np_idx][6] == 0:
                new_list[id][np_idx][6] = row.loc[row.index[0]]['exp']
                if row.loc[row.index[0]]['exp'] == 0:
                    w.warn(mess.format('exp'))
            if new_list[id][np_idx][8] == 0:
                new_list[id][np_idx][8] = row.loc[row.index[0]]['slopes']
                if row.loc[row.index[0]]['slopes'] == 0:
                    w.warn(mess.format('slopes'))
            if new_list[id][np_idx][12] == 0:
                sum = 0
                for i in range(len(row.index)):
                    sum += row.loc[row.index[i]]['increment']
                new_list[id][np_idx][12] = sum / len(row.index)
                if new_list[id][np_idx][12] == 0:
                    w.warn(mess.format('increment'))
                del sum
            if new_list[id][np_idx][13] == 0:
                count = 0
                for i in range(len(row.index)):
                    # could at break statement to make more efficient, but that would involve handling > and < cases
                    if row.loc[row.index[i]]['tp'] == idx[1]:
                        count += 1
                new_list[id][np_idx][13] = count
                if count == 0:
                    w.warn(mess.format('tp'))
            del row, mess
        for new_dict in process_dict(val['Actual Returns']):
            id = str(new_dict['idea'])
            id_set.add(id)
            new_list = check_id(new_list, id, param_size, tp_active)
            # scaling total actual returns in a TP to that in a loop (more accurate for smart_returns() in optimize.py)
            # actual_returns_tp / times invested --> makes sense because of average increment
            new_list[id][np_idx][14] += new_dict['returns'] / new_list[id][np_idx][13]
    # run for last scientist
    if new_list != {}:  # check that row was not empty
        # for nl in list(new_list.values()):  # a list of np arrays
        append_data = np.asarray(list(new_list.values()))  # 3D array
        if current_data is None:
            current_data = from_3d_to_2d(append_data)
        else:
            # print(current_data.shape, append_data.shape)
            current_data = np.concatenate((current_data, from_3d_to_2d(append_data)))

    if has_past:
        past_data = np.load('tmp/big_data.npy')
        np.save('tmp/big_data.npy', np.concatenate((past_data, current_data)))
    else:
        np.save('tmp/big_data.npy', current_data)  # np.concatenate((past_data, current_data)))
    print('Big data now has {} elements'.format(len(np.load('tmp/big_data.npy'))))


if __name__ == '__main__':
    main()
