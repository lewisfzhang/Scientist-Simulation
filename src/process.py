# process.py

from store import *
from functions import *


# assign agent returns and updating agent df
def process_winners_equal(model):
    model.investing_queue = get_investing_queue(model)
    model.actual_returns_matrix = unlock_actual_returns(model, None)
    unpack_model_lists(model, None)

    # set that stores scientists who get returns (optional, see below)
    happy_scientist = set()

    # counts number of scientists who invested in the idea in this period
    num_scientists = np.zeros(model.total_ideas)

    # keeps track of which ideas have already been iterated
    idea_actual = np.zeros(model.total_ideas)

    # iterating through all investments by each scientists
    for index, row in model.investing_queue.iterrows():
        idx_idea = int(row['Idea Choice'])
        num_scientists[idx_idea] += 1
        stop_index = int(model.total_effort[idx_idea])
        start_index = int(model.total_effort_start[idx_idea])
        if idea_actual[idx_idea] == 0:  # if the idea has not been computed yet
            actual_return = get_returns(idx_idea, model.actual_returns_matrix, start_index, stop_index)
            model.total_actual_returns[idx_idea] += actual_return
            idea_actual += actual_return
            del actual_return
        row['Actual Return'] = (row['Marginal'] / (stop_index - start_index)) * idea_actual[idx_idea]
        idx_id = int(row["ID"])
        happy_scientist.add(idx_id)
        model.schedule.agents[model.agent_dict[idx_id]].update(row)

        del stop_index, start_index, idx_idea, index, row, idx_id

    for idea in range(len(num_scientists)):
        if num_scientists[idea] != 0:  # at least one scientist invested during this TP
            model.exp_bayes[idea].append(num_scientists[idea])
        del idea

    # updating data for remaining scientists who did not get any returns / inactive scientists
    # NOTE: OPTIONAL, only for agent_df. can speed up model simulation greatly if we don't care about
    # agent df with NaN for active scientists (happy_scientists would also be optional in this case)
    # but then you wouldn't be able to see how much all scientists invested in this period
    if model.schedule.time >= 2:
        if config.all_scientists:
            range_list = range(1, model.num_scientists + 1)
        else:
            interval = model.schedule.time - config.time_periods_alive
            if interval < 0:
                interval = 0
            range_list = range(1 + interval * config.N, model.schedule.time * config.N + 1)
            del interval
        for i in list(set(range_list) - happy_scientist):  # difference between two sets
            model.schedule.agents[model.agent_dict[i]].update(None)
        del range_list

    store_actual_returns(model, None)
    store_model_lists(model, False, None)
    del model.investing_queue, happy_scientist, num_scientists, idea_actual


# assign agent returns and updating agent df
def process_winners_old(model):
    model.investing_queue = get_investing_queue(model)
    model.actual_returns_matrix = unlock_actual_returns(model, None)
    unpack_model_lists(model, None)

    # initializing list of dictionaries with returns and investments for each idea in the TP
    list_dict = new_list_dict(model)

    # iterating through all investments by each scientists
    # young scientists get 0 returns, old scientists get all of the returns
    for index, row in model.investing_queue.iterrows():
        idx_idea = int(row['Idea Choice'])
        # NOTE: OLDER SCIENTIST HAS THE YOUNGER UNIQUE ID!!!
        if list_dict[idx_idea]['Oldest ID'] > row['ID']:
            # -1 shifts for zero-based array
            idx_id_old = int(list_dict[idx_idea]['Oldest ID']) - 1
            idx_id_new = int(row['ID']) - 1
            if idx_id_old != model.num_scientists:  # only active ideas
                # NOTE: for old scientist we already appended so changing appended value from 1 to 0
                model.exp_bayes[idx_id_old][idx_idea][len(model.exp_bayes[idx_id_old][idx_idea]) - 1] = 0
            model.exp_bayes[idx_id_new][idx_idea].append(1)
            list_dict[idx_idea]['Oldest ID'] = row['ID']
            del idx_id_old, idx_id_new

        # update current stats for list_dict
        if list_dict[idx_idea]["Updated"] is False:
            stop_index = int(model.total_effort[idx_idea])
            start_index = int(model.total_effort_start[idx_idea])

            actual_return = get_returns(idx_idea, model.actual_returns_matrix, start_index, stop_index)

            list_dict[idx_idea]['Actual Return'] += actual_return
            model.total_actual_returns[idx_idea] += actual_return

            list_dict[idx_idea]["Updated"] = True  # so it never runs again
            del stop_index, start_index

        # max (perceived) return is not as accurate since scientists don't know
        # what others are investing --> only can account for in actual return
        list_dict[idx_idea]['Max Return'] += row['Max Return']

        del index, row, idx_idea

    # set that stores scientists who get returns (optional, see below)
    happy_scientist = set()

    # update model data collecting variables, and back for the old scientist who won all the returns for the idea
    for idx, idea_choice in enumerate(list_dict):
        idx_id = int(idea_choice["Oldest ID"])
        if idx_id != model.num_scientists + 1:  # only active ideas needed
            happy_scientist.add(idx_id)
            model.schedule.agents[model.agent_dict[idx_id]].update(idea_choice)
        del idx, idea_choice, idx_id

    # updating data for remaining scientists who did not get any returns / inactive scientists
    # NOTE: OPTIONAL, only for agent_df. can speed up model simulation greatly if we don't care about
    # agent df with NaN for active scientists (happy_scientists would also be optional in this case)
    # but then you wouldn't be able to see how much all scientists invested in this period
    if model.schedule.time >= 2:
        if config.all_scientists:
            range_list = range(1, model.num_scientists + 1)
        else:
            interval = model.schedule.time - config.time_periods_alive
            if interval < 0:
                interval = 0
            range_list = range(1 + interval * config.N, model.schedule.time * config.N + 1)
            del interval
        for i in list(set(range_list) - happy_scientist):  # difference between two sets
            model.schedule.agents[model.agent_dict[i]].update(None)
        del range_list

    store_actual_returns(model, None)
    store_model_lists(model, False, None)
    del model.investing_queue, list_dict, happy_scientist
