# final.py

from functions import *
from store import *


# for data collecting after model has finished running
def collect_vars(model):
    f_print("\n\ndone with step 9")
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
    idea = range(0, model.total_ideas, 1)
    tp = np.arange(model.total_ideas) // config.ideas_per_time
    prop_invested = rounded_tuple(model.total_effort / (config.true_means_lam + 3 * config.true_sds_lam))
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
    np.save(model.directory + 'idea_phase.npy', idea_phase)
    store_model_arrays_data(model, False, None)
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
                          flatten_list(model.final_actual_returns_invested_ideas))}
    pd.DataFrame.from_dict(ind_ideas_dict).to_pickle(model.directory + 'ind_ideas.pkl')
    if config.use_store_model is False:
        with open(model.directory + "final_perceived_returns_invested_ideas.txt", "wb") as fp:
            pickle.dump(model.final_perceived_returns_invested_ideas, fp)
    store_model_lists(model, False, None)
    del ind_ideas_dict
    # </editor-fold>

    # <editor-fold desc="Part 3: social_output, ideas_entered">
    ind_vars = pd.read_pickle(model.directory + 'ind_ideas.pkl')
    actual_returns = agent_vars[agent_vars['Actual Returns'].str.startswith("{", na=False)]['Actual Returns']

    # format of scientist_tracker: [agent_id][num_ideas_invested][total_returns]
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
    np.save(model.directory + 'social_output.npy', np.asarray(y_var))
    np.save(model.directory + 'ideas_entered.npy', np.asarray(x_var))
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
