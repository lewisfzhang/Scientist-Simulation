# store.py
# manages all store and access functions for npy and list

import numpy as np
import pickle
import config
import pandas as pd
from collections import Counter


def store_model_arrays(model, is_first, lock):
    if config.use_store:
        if is_first:
            np.save(model.directory + 'idea_periods.npy', model.idea_periods)
        del model.idea_periods

    if not is_first and config.use_multiprocessing and lock is not None:
        lock.release()


def store_model_arrays_data(model, is_first, lock):
    if config.use_store:
        np.save(model.directory + 'total_effort.npy', model.total_effort)
        model.total_effort = None

        np.save(model.directory + 'effort_invested_by_age.npy', model.effort_invested_by_age)
        model.effort_invested_by_age = None

        np.save(model.directory + 'total_perceived_returns.npy', model.total_perceived_returns)
        model.total_perceived_returns = None

        np.save(model.directory + 'total_actual_returns.npy', model.total_actual_returns)
        model.total_actual_returns = None

        np.save(model.directory + 'total_k.npy', model.total_k)
        model.total_k = None

        np.save(model.directory + 'total_times_invested.npy', model.total_times_invested)
        model.total_times_invested = None

        np.save(model.directory + 'total_scientists_invested.npy', model.total_scientists_invested)
        model.total_scientists_invested = None

        np.save(model.directory + 'total_scientists_invested_helper.npy', model.total_scientists_invested_helper)
        model.total_scientists_invested_helper = None

        np.save(model.directory + 'total_idea_phase.npy', model.total_idea_phase)
        model.total_idea_phase = None

        np.save(model.directory + 'idea_phase_label.npy', model.idea_phase_label)
        model.idea_phase_label = None

    if not is_first and config.use_multiprocessing and lock is not None:
        lock.release()


def store_model_lists(model, is_first, lock):
    if config.use_store:
        with open(model.directory + "final_perceived_returns_invested_ideas.txt", "wb") as fp:
            pickle.dump(model.final_perceived_returns_invested_ideas, fp)
        model.final_perceived_returns_invested_ideas = None

        with open(model.directory + "final_k_invested_ideas.txt", "wb") as fp:
            pickle.dump(model.final_k_invested_ideas, fp)
        model.final_k_invested_ideas = None

        with open(model.directory + "final_actual_returns_invested_ideas.txt", "wb") as fp:
            pickle.dump(model.final_actual_returns_invested_ideas, fp)
        model.final_actual_returns_invested_ideas = None

        with open(model.directory + "final_idea_idx.txt", "wb") as fp:
            pickle.dump(model.final_idea_idx, fp)
        model.final_idea_idx = None

        with open(model.directory + "final_scientist_id.txt", "wb") as fp:
            pickle.dump(model.final_scientist_id, fp)
        model.final_scientist_id = None

        with open(model.directory + "final_marginal_invested_ideas.txt", "wb") as fp:
            pickle.dump(model.final_marginal_invested_ideas, fp)
        model.final_marginal_invested_ideas = None

        with open(model.directory + "final_slope.txt", "wb") as fp:
            pickle.dump(model.final_slope, fp)
        model.final_slope = None

        if is_first:
            np.save(model.directory + 'actual_returns_matrix.npy', model.actual_returns_matrix)
        model.actual_returns_matrix = None

    if not is_first and config.use_multiprocessing and lock is not None:
        lock.release()
  

def unpack_model_arrays(model, lock):
    if config.use_multiprocessing and lock is not None:
        lock.acquire()
    if config.use_store:
        # unlimited access to past ideas, too lazy to think of another way to implement double negative
        # what this statement really wants is idea_periods <= schedule.time
        model.idea_periods = np.load(model.directory + "idea_periods.npy")


def unpack_model_arrays_data(model, lock):
    if config.use_multiprocessing and lock is not None:
        lock.acquire()
    if config.use_store:
        model.total_effort = np.load(model.directory + 'total_effort.npy')

        model.effort_invested_by_age = np.load(model.directory + 'effort_invested_by_age.npy')

        model.total_perceived_returns = np.load(model.directory + 'total_perceived_returns.npy')

        model.total_actual_returns = np.load(model.directory + 'total_actual_returns.npy')

        model.total_k = np.load(model.directory + 'total_k.npy')

        model.total_times_invested = np.load(model.directory + 'total_times_invested.npy')

        model.total_scientists_invested = np.load(model.directory + 'total_scientists_invested.npy')

        model.total_scientists_invested_helper = np.load(model.directory + 'total_scientists_invested_helper.npy')

        model.total_idea_phase = np.load(model.directory + 'idea_phase.npy')

        model.idea_phase_label = np.load(model.directory + 'idea_phase.npy')



def unpack_model_lists(model, lock):
    if config.use_multiprocessing and lock is not None:
        lock.acquire()
    if config.use_store:
        with open(model.directory + "final_perceived_returns_invested_ideas.txt", "rb") as fp:
            model.final_perceived_returns_invested_ideas = pickle.load(fp)

        with open(model.directory + "final_k_invested_ideas.txt", "rb") as fp:
            model.final_k_invested_ideas = pickle.load(fp)

        with open(model.directory + "final_actual_returns_invested_ideas.txt", "rb") as fp:
            model.final_actual_returns_invested_ideas = pickle.load(fp)

        with open(model.directory + "final_scientist_id.txt", "rb") as fp:
            model.final_scientist_id = pickle.load(fp)

        with open(model.directory + "final_idea_idx.txt", "rb") as fp:
            model.final_idea_idx = pickle.load(fp)

        with open(model.directory + "final_marginal_invested_ideas.txt", "rb") as fp:
            model.final_marginal_invested_ideas = pickle.load(fp)

        with open(model.directory + "final_slope.txt", "rb") as fp:
            model.final_slope = pickle.load(fp)


def unlock_actual_returns(model, lock):
    if config.use_multiprocessing and lock is not None:
        lock.acquire()
    if config.use_store:
        model.actual_returns_matrix = np.load(model.directory + 'actual_returns_matrix.npy')
    return model.actual_returns_matrix


def store_actual_returns(model, lock):
    if config.use_store:
        model.actual_returns_matrix = None
    if config.use_multiprocessing and lock is not None:
        lock.release()


def create_datacollectors(model):
    # for agent
    index = pd.MultiIndex.from_product([range(0, config.time_periods + 2), range(1, model.num_scientists + 1)],
                                       names=['Step', 'AgentID'])
    columns = ['TP Born', 'Effort Invested In Period (K)', 'Effort Invested In Period (Marginal)',
               'Perceived Returns', 'Actual Returns']
    if config.use_store:
        pd.DataFrame(index=index, columns=columns).to_pickle(model.directory+'agent_vars_df.pkl')
    else:
        model.agent_df = pd.DataFrame(index=index, columns=columns)

    # for model
    index = range(config.time_periods + 2)
    columns = ['Total Effort List', 'Total Effort By Age']
    if config.use_store:
        pd.DataFrame(index=index, columns=columns).to_pickle(model.directory+'model_vars_df.pkl')
    else:
        model.model_df = pd.DataFrame(index=index, columns=columns)

    del index, columns
    

def store_agent_arrays(agent):
    if config.use_store:
        np.save(agent.directory+'perceived_returns_matrix.npy', agent.perceived_returns_matrix)
        agent.perceived_returns_matrix = None

        np.save(agent.directory + 'k.npy', agent.k)
        agent.k = None


def store_agent_arrays_tp(agent):
    if config.use_store:
        np.save(agent.directory + 'marginal_invested_by_scientist.npy', agent.marginal_invested_by_scientist)
        agent.marginal_invested_by_scientist = None

        np.save(agent.directory + 'k_invested_by_scientist.npy', agent.k_invested_by_scientist)
        agent.k_invested_by_scientist = None

        np.save(agent.directory + 'eff_inv_in_period_k.npy', agent.eff_inv_in_period_k)
        agent.eff_inv_in_period_k = None

        np.save(agent.directory + 'eff_inv_in_period_marginal.npy', agent.eff_inv_in_period_marginal)
        agent.eff_inv_in_period_marginal = None

        np.save(agent.directory + 'perceived_returns_tp.npy', agent.perceived_returns_tp)
        agent.perceived_returns_tp = None

        np.save(agent.directory + 'actual_returns_tp.npy', agent.actual_returns_tp)
        agent.actual_returns_tp = None


def unpack_agent_arrays(agent):
    if config.use_store:
        agent.perceived_returns_matrix = np.load(agent.directory + 'perceived_returns_matrix.npy')

        agent.k = np.load(agent.directory + 'k.npy')


def unpack_agent_arrays_tp(agent):
    if config.use_store:
        agent.eff_inv_in_period_k = np.load(agent.directory + 'eff_inv_in_period_k.npy')

        agent.eff_inv_in_period_marginal = np.load(agent.directory + 'eff_inv_in_period_marginal.npy')

        agent.marginal_invested_by_scientist = np.load(agent.directory + 'marginal_invested_by_scientist.npy')

        agent.k_invested_by_scientist = np.load(agent.directory + 'k_invested_by_scientist.npy')

        agent.perceived_returns_tp = np.load(agent.directory + 'perceived_returns_tp.npy')

        agent.actual_returns_tp = np.load(agent.directory + 'actual_returns_tp.npy')


def create_list_dict(model):
    list_dict = []
    for i in range(model.total_ideas):
        # Counter class can be treated as a dictionary, but useful for adding dicts
        # NOTE: model.num_scientists+1 is one greater than the 'greatest'/'last' scientist's id
        idea_dict = Counter({"Idea Choice": i, "Max Return": 0, "Actual Return": 0,
                             "Oldest ID": model.num_scientists+1, "Updated": False})
        list_dict.append(idea_dict)
        del idea_dict

    with open(model.directory + "list_dict.txt", "wb") as fp:
        pickle.dump(list_dict, fp)
    del list_dict


def new_list_dict(model):
    with open(model.directory + "list_dict.txt", "rb") as fp:
        return pickle.load(fp)


def update_investing_queue(model, df_data, lock):
    if config.use_multiprocessing:
        lock.acquire()
    if config.use_store:
        investing_queue = pd.read_pickle(model.directory + 'investing_queue.pkl')
        investing_queue = investing_queue.append(df_data, ignore_index=True)
        investing_queue.to_pickle(model.directory + 'investing_queue.pkl')
        del investing_queue
    else:
        model.investing_queue = model.investing_queue.append(df_data, ignore_index=True)
    if config.use_multiprocessing:
        lock.release()


def get_investing_queue(model):
    if config.use_store:
        model.investing_queue = pd.read_pickle(model.directory + 'investing_queue.pkl')
    return model.investing_queue


def new_investing_queue(model):
    if config.use_store:
        # queue format: idea_choice, scientist.marginal_effort[idea_choice], increment, max_return,
        #               actual_return, scientist.unique_id
        pd.DataFrame(columns=['Idea Choice', 'Max Return', 'ID']).to_pickle(model.directory + 'investing_queue.pkl')
    else:
        model.investing_queue = pd.DataFrame(columns=['Idea Choice', 'Max Return', 'ID'])


def get_total_start_effort(model):
    model.total_effort_start = np.load(model.directory + 'total_effort.npy') if config.use_store \
        else np.copy(model.total_effort)


def copy_total_start_effort(scientist, lock):
    if config.use_multiprocessing and lock is not None:  # corresponding checks if we are using multiprocessing
        lock.acquire()
    scientist.total_effort_start = np.copy(scientist.model.total_effort_start)
    if config.use_multiprocessing and lock is not None:
        lock.release()


def update_agent_df(scientist, new_data):
    if config.use_store:
        # updating agent dataframe
        df_agent = pd.read_pickle(scientist.model.directory + 'agent_vars_df.pkl')
        df_agent.loc[scientist.model.schedule.time].loc[scientist.unique_id] = new_data
        df_agent.to_pickle(scientist.model.directory + 'agent_vars_df.pkl')
        del df_agent
    else:
        scientist.model.agent_df.loc[scientist.model.schedule.time].loc[scientist.unique_id] = new_data


def update_model_df(model, data_list):
    if config.use_store:
        df_model = pd.read_pickle(model.directory + 'model_vars_df.pkl')
        df_model.loc[model.schedule.time] = data_list
        df_model.to_pickle(model.directory + 'model_vars_df.pkl')
        del df_model
    else:
        model.model_df.loc[model.schedule.time] = data_list
