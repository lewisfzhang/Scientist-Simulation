# store.py
# manages all store and access functions for npy and list

import numpy as np
import pickle
import input_file
import pandas as pd
from collections import Counter
import shared_mp as s


def store_model_arrays(model, isFirst):
    np.save(model.directory + 'idea_periods.npy', model.idea_periods)
    model.idea_periods = None

    if not isFirst and input_file.use_multiprocessing:
        s.lock1.release()


def store_model_arrays_data(model, isFirst):
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

    if not isFirst and input_file.use_multiprocessing:
        s.lock2.release()


def store_model_lists(model, isFirst):
    with open(model.directory + "final_perceived_returns_invested_ideas.txt", "wb") as fp:
        pickle.dump(model.final_perceived_returns_invested_ideas, fp)
    model.final_perceived_returns_invested_ideas = None

    with open(model.directory + "final_k_invested_ideas.txt", "wb") as fp:
        pickle.dump(model.final_k_invested_ideas, fp)
    model.final_k_invested_ideas = None

    with open(model.directory + "final_actual_returns_invested_ideas.txt", "wb") as fp:
        pickle.dump(model.final_actual_returns_invested_ideas, fp)
    model.final_actual_returns_invested_ideas = None

    np.save(model.directory + 'actual_returns_matrix.npy', model.actual_returns_matrix)
    model.actual_returns_matrix = None

    if not isFirst and input_file.use_multiprocessing:
        s.lock3.release()
  
    
def unpack_model_arrays(model):
    if input_file.use_multiprocessing:
        s.lock1.acquire()

    # unlimited access to past ideas, too lazy to think of another way to implement double negative
    # what this statement really wants is idea_periods <= schedule.time
    model.idea_periods = np.load(model.directory + "idea_periods.npy")


def unpack_model_arrays_data(model):
    if input_file.use_multiprocessing:
        s.lock2.acquire()

    model.total_effort = np.load(model.directory + 'total_effort.npy')

    model.effort_invested_by_age = np.load(model.directory + 'effort_invested_by_age.npy')

    model.total_perceived_returns = np.load(model.directory + 'total_perceived_returns.npy')

    model.total_actual_returns = np.load(model.directory + 'total_actual_returns.npy')

    model.total_k = np.load(model.directory + 'total_k.npy')

    model.total_times_invested = np.load(model.directory + 'total_times_invested.npy')

    model.total_scientists_invested = np.load(model.directory + 'total_scientists_invested.npy')


def unpack_model_lists(model):
    if input_file.use_multiprocessing:
        s.lock3.acquire()

    with open(model.directory + "final_perceived_returns_invested_ideas.txt", "rb") as fp:
        model.final_perceived_returns_invested_ideas = pickle.load(fp)

    with open(model.directory + "final_k_invested_ideas.txt", "rb") as fp:
        model.final_k_invested_ideas = pickle.load(fp)

    with open(model.directory + "final_actual_returns_invested_ideas.txt", "rb") as fp:
        model.final_actual_returns_invested_ideas = pickle.load(fp)

    model.actual_returns_matrix = np.load(model.directory + 'actual_returns_matrix.npy')


def create_datacollectors(model):
    # for agent
    index = pd.MultiIndex.from_product([range(0,input_file.time_periods+2), range(1, model.num_scientists+1)],
                                      names=['Step', 'AgentID'])
    columns = ['TP Born', 'Total Effort Invested', 'Effort Invested In Period (Increment)',
               'Effort Invested In Period (Marginal)', 'Perceived Returns', 'Actual Returns']
    pd.DataFrame(index=index, columns=columns).to_pickle(model.directory+'agent_vars_df.pkl')

    # for model
    index = range(input_file.time_periods+2)
    columns = ['Total Effort List', 'Total Effort By Age']
    pd.DataFrame(index=index, columns=columns).to_pickle(model.directory+'model_vars_df.pkl')

    del index, columns
    

def store_agent_arrays(agent):
    np.save(agent.directory+'perceived_returns_matrix.npy', agent.perceived_returns_matrix)
    agent.perceived_returns_matrix = None

    np.save(agent.directory + 'k.npy', agent.k)
    agent.k = None


def store_agent_arrays_data(agent):
    np.save(agent.directory + 'perceived_returns.npy', agent.perceived_returns)
    agent.perceived_returns = None

    np.save(agent.directory + 'actual_returns.npy', agent.actual_returns)
    agent.actual_returns = None


def store_agent_arrays_tp(agent):
    np.save(agent.directory + 'effort_invested_by_scientist.npy', agent.effort_invested_by_scientist)
    agent.effort_invested_by_scientist = None

    np.save(agent.directory + 'eff_inv_in_period_increment.npy', agent.eff_inv_in_period_increment)
    agent.eff_inv_in_period_increment = None

    np.save(agent.directory + 'eff_inv_in_period_marginal.npy', agent.eff_inv_in_period_marginal)
    agent.eff_inv_in_period_marginal = None


def unpack_agent_arrays(agent):
    agent.perceived_returns_matrix = np.load(agent.directory + 'perceived_returns_matrix.npy')

    agent.k = np.load(agent.directory + 'k.npy')


def unpack_agent_arrays_data(agent):
    agent.perceived_returns = np.load(agent.directory + 'perceived_returns.npy')

    agent.actual_returns = np.load(agent.directory + 'actual_returns.npy')


def unpack_agent_arrays_tp(agent):
    agent.eff_inv_in_period_increment = np.load(agent.directory + 'eff_inv_in_period_increment.npy')

    agent.eff_inv_in_period_marginal = np.load(agent.directory + 'eff_inv_in_period_marginal.npy')

    agent.effort_invested_by_scientist = np.load(agent.directory + 'effort_invested_by_scientist.npy')


def create_list_dict(model):
    list_dict = []
    for i in range(model.total_ideas):
        # Counter class can be treated as a dictionary, but useful for adding dicts
        idea_dict = Counter({"Idea Choice": i, "Max Return": 0, "Actual Return": 0, "Oldest ID": 0, "Total Effort": 0,
                             "Total Increment": 0})
        list_dict.append(idea_dict)

    with open(model.directory + "list_dict.txt", "wb") as fp:
        pickle.dump(list_dict, fp)

    list_dict = None
    idea_dict = None


def new_list_dict(model):
    with open(model.directory + "list_dict.txt", "rb") as fp:
        return pickle.load(fp)