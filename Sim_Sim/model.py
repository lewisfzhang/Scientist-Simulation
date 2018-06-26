from mesa import Agent, Model
from mesa.time import BaseScheduler
import numpy as np
from numpy.random import poisson
from functions import *  # anything not directly tied to Mesa objects
from optimize import *
import math
import timeit
import pandas as pd
import multiprocessing as mp
import os
import input_file
import pickle
from collections import Counter
from store import *
import shared_mp as s


# utility = important, don't run GC on it
# void = not important, can run GC

# This file has 3 parts
# 1. Agent class from Mesa
# 2. Model class from Mesa
class Scientist(Agent):
    # scalars initialized first, then arrays
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # Allows scientists to access model variables
        self.model = model

        # Scalar: amount of effort a scientist starts with in each time period
        # he or she is alive (not accounting for start effort decay)
        self.start_effort = poisson(lam=input_file.start_effort_lam)  # utility

        # Scalar: amount of effort a scientist has left; goes down within a
        # given time period as a scientist invests in various ideas
        self.avail_effort = self.start_effort  # utility

        # the specific agent id unique to him/herself
        self.unique_id = unique_id  # utility

        # SCALAR: Each scientist is assigned a unique ID. We also use unique IDs to determine
        # scientists' ages and thus which scientists are alive in a given time period and which ideas they can invest in
        self.birth_order = math.ceil(unique_id / input_file.N)  # utility

        # Check a scientist's age in the current time period
        # NOTE: model time begins at 0
        self.current_age = self.model.schedule.time - self.birth_order  # utility

        # create specific tmp directory for agent
        self.directory = 'tmp/agent_' + str(self.unique_id) + '/'
        create_directory(self.directory)

        # Array: investment cost for each idea for a given scientist; a scientist
        # must first pay an idea's investment cost before receiving returns from
        # additional investment (so each scientist has different cost for each idea)
        #
        # normal distribution is the "noise" associated with learning an idea where 99% of
        # all true k for each scientist will be within 50% of the true k based on each idea
        self.k = self.model.k * (np.random.normal(6, 1, self.model.total_ideas)/6)  # utility

        # Array: has length equal to the total number of ideas in the model,
        # with 0s, 1s, 2s, etc. that indicate which time periods ideas are from
        self.idea_periods = np.arange(self.model.total_ideas) // input_file.ideas_per_time  # utility

        # ARRAY: error each scientist has for perceived compared to the true actual returns
        self.noise = input_file.noise_factor * np.random.normal(0, 1, model.total_ideas)  # void

        # Arrays: parameters determining perceived returns for ideas, which are
        # distinct from true returns. Ideas are modeled as logistic CDFs ("S" curve)
        self.sds = self.model.true_sds + self.noise  # void
        self.means = self.model.true_means + self.noise  # void

        # ARRAY: Create the ideas/returns matrix
        # NOTE: logistic_cdf is not ranodm, always generates same curve based on means and sds
        self.perceived_returns_matrix = create_return_matrix(self.model.total_ideas, self.sds, self.means, self.model.M,
                                                             input_file.true_sds_lam, input_file.true_means_lam)  # utility

        # dereferencing 'void' variables
        self.noise = None
        self.sds = None
        self.means = None

        # Array: keeps track of how much effort a scientist has invested in each idea
        # NOTE: does NOT include effort paid for ideas' investment costs
        self.effort_invested_by_scientist = np.zeros(self.model.total_ideas)  # utility

        # Array: keeps track of total, perceived returns for each idea given
        # a scientist's level of investment in that idea
        self.perceived_returns = np.zeros(self.model.total_ideas)  # utility
        self.actual_returns = np.zeros(self.model.total_ideas)  # utility

        # Array: keeps track of effort invested ONLY during the current time period
        # NOTE: resets to 0 after each time period and DOES include investment costs
        self.eff_inv_in_period_increment = np.zeros(self.model.total_ideas)  # utility
        self.eff_inv_in_period_marginal = np.zeros(self.model.total_ideas)  # utility

        store_agent_arrays(self)
        store_agent_arrays_data(self)
        store_agent_arrays_tp(self)

    def step(self):
        # Check a scientist's age in the current time period
        # NOTE: model time begins at 0
        self.current_age = self.model.schedule.time - self.birth_order

        # scientist is alive
        if 0 <= self.current_age < input_file.time_periods_alive and self.model.schedule.time >= 2:
            # pull all variables out of stored files
            unpack_agent_arrays(self)
            unpack_agent_arrays_tp(self)

            # reset effort in new time period
            self.eff_inv_in_period_increment[:] = 0
            self.eff_inv_in_period_marginal[:] = 0

            # reset effort
            self.avail_effort = self.start_effort

            unpack_model_arrays(self.model)
            # Array: keeps track of which ideas can be worked on in a given time period
            # for a given scientist's age. This array will have 1s for ideas that
            # are accessible to a scientist and 0s for all other ideas
            self.avail_ideas = np.logical_not(self.model.idea_periods > self.model.schedule.time)
            store_model_arrays(self.model, False)

            greedy_investing(self)

            self.avail_ideas = None
            store_agent_arrays(self)
            store_agent_arrays_tp(self)

        else:
            unpack_agent_arrays_data(self)
            unpack_agent_arrays_tp(self)

            if self.current_age == input_file.time_periods_alive:
                # reset effort in new time period
                self.eff_inv_in_period_increment[:] = 0
                self.eff_inv_in_period_marginal[:] = 0

                # reset returns because scientists is not active
                self.perceived_returns[:] = 0
                self.actual_returns[:] = 0

            # updating 0's to agent_df
            # comment out to get more concise agent df that only displays active scientists in a TP
            # self.update_agent_df()

            store_agent_arrays_tp(self)
            store_agent_arrays_data(self)

        gc_collect()

    # converts mutable numpy arrays into easily accessed tuples
    def update(self, df_row):
        unpack_agent_arrays_data(self)
        unpack_agent_arrays_tp(self)

        if df_row is not None:
            # Updates parameters after idea selection and effort expenditure
            # NOTE: self.avail_effort and self.eff_inv_in_period should be
            # updated by the increment, not by marginal effort, because the
            # increment includes investment costs. We don't care about
            # paid investment costs for the other variables
            idea = int(df_row['Idea Choice'])
            self.perceived_returns[idea] += df_row['Max Return']
            self.actual_returns[idea] += df_row['Actual Return']
            idea = None

        self.update_agent_df()

        store_agent_arrays_data(self)
        store_agent_arrays_tp(self)

    def update_agent_df(self):
        # updating agent dataframe
        df_agent = pd.read_pickle('tmp/model/agent_vars_df.pkl')
        # format: TP Born, Total effort invested, Effort invested in period (increment),
        #         Effort invested in period (marginal), Perceived returns, Actual returns
        new_data = {'TP Born': self.birth_order,
                    'Total Effort Invested': self.effort_invested_by_scientist,
                    'Effort Invested In Period (Increment)': self.eff_inv_in_period_increment,
                    'Effort Invested In Period (Marginal)': self.eff_inv_in_period_marginal,
                    'Perceived Returns': rounded_tuple(self.perceived_returns),
                    'Actual Returns': rounded_tuple(self.actual_returns)}
        df_agent.loc[self.model.schedule.time].loc[self.unique_id] = new_data
        df_agent.to_pickle(self.model.directory+'agent_vars_df.pkl')

        # Dereferencing variables
        df_agent = None
        new_data = None


class ScientistModel(Model):
    def __init__(self, seed=None):
        super().__init__(seed)

        # create specific tmp directory for model
        create_directory('tmp/')
        self.directory = 'tmp/model/'
        create_directory(self.directory)

        # Scalar: indicates the total number of scientists in the model
        self.num_scientists = input_file.N*(input_file.time_periods + 1)  # utility

        # Scalar: total number of ideas in the model. +2 is used to account
        # for first two, non-steady state time periods
        self.total_ideas = input_file.ideas_per_time*(input_file.time_periods+2)  # utility

        # Array: has length equal to the total number of ideas in the model,
        # with 0s, 1s, 2s, etc. that indicate which time periods ideas are from
        self.idea_periods = np.arange(self.total_ideas) // input_file.ideas_per_time  # utility

        # k is the learning cost for each idea
        self.k = poisson(lam=input_file.k_lam, size=self.total_ideas)  # void/utility (used for init agent objects)

        # Array: store parameters for true idea return distribution
        self.true_sds = poisson(lam=input_file.true_sds_lam, size=self.total_ideas)  # void
        self.true_means = poisson(lam=input_file.true_means_lam, size=self.total_ideas)  # void

        # Ensures that none of the standard devs are equal to 0, this is OKAY
        self.true_sds += 1 + input_file.noise_factor  # void

        # M is a scalar that multiples based on each idea
        # not sure if this is redundant since we already have random poisson values for true_means and true_sds
        self.M = poisson(lam=10000, size=self.total_ideas)  # void

        # creates actual returns matrix
        self.actual_returns_matrix = create_return_matrix(self.total_ideas, self.true_sds, self.true_means, self.M,
                                                          input_file.true_sds_lam, input_file.true_means_lam)  # utility

        # Array: keeps track of total effort allocated to each idea across all scientists
        self.total_effort = np.zeros(self.total_ideas)

        # data collector variable
        # format = [young,old]
        self.effort_invested_by_age = [np.zeros(self.total_ideas), np.zeros(self.total_ideas)]

        # data collector variables that have no effect on results of model
        self.total_perceived_returns = np.zeros(self.total_ideas)
        self.total_actual_returns = np.zeros(self.total_ideas)
        self.total_k = np.zeros(self.total_ideas)
        self.total_times_invested = np.zeros(self.total_ideas)
        self.total_scientists_invested = np.zeros(self.total_ideas)

        # Array: keeping track of all the returns of investing in each available and invested ideas
        # NOTE: the K is based on initial learning cost, not current cost
        self.final_perceived_returns_invested_ideas = [[] for i in range(self.num_scientists)]
        self.final_k_invested_ideas = []
        self.final_actual_returns_invested_ideas = []

        # Make scientists choose ideas and allocate effort in a random order
        # for each step of the model (i.e. within a time period, the order
        # in which young and old scientists get to invest in ideas is random)
        self.schedule = BaseScheduler(self)  # NOTE: doesn't skew results if not random due to call_back

        # creates Agent objects
        # p = mp.Pool()
        # scientists = p.starmap(scientist_creator, [(i, self) for i in range(1, self.num_scientists + 1)])
        # p.close()
        # p.join()

        # adds Agent objects to the schedule
        for i in range(1, self.num_scientists + 1):
            self.schedule.add(Scientist(i, self))

        # dereferencing variables
        self.k = None
        self.true_sds = None
        self.true_means = None
        self.M = None

        create_datacollectors(self)
        create_list_dict(self)
        store_model_arrays(self, True)
        store_model_arrays_data(self, True)
        store_model_lists(self, True)

    def step(self):
        # queue format: idea_choice, scientist.marginal_effort[idea_choice], increment, max_return,
        #               actual_return, scientist.unique_id
        pd.DataFrame(columns=['Idea Choice', 'Marginal Effort', 'Increment', 'Max Return', 'Actual Return', 'ID',
                              'Times Invested']).to_pickle('tmp/model/investing_queue.pkl')

        # iterates through all scientists in the model
        # below is the same as self.schedule.step()
        if input_file.use_multiprocessing:
            s.lock1 = mp.Lock()  # for model_arrays
            s.lock2 = mp.Lock()  # for model_arrays_data
            s.lock3 = mp.Lock()  # for model_lists
            s.lock4 = mp.Lock()  # for model df
            p = mp.Pool()
            p.starmap(mp_helper, [(self, i) for i in range(self.num_scientists)])
            p.close()
            p.join()
            p = None
        else:
            for i in range(self.num_scientists):
                self.schedule.agents[i].step()

        self.call_back()
        self.schedule.time += 1

        # run data collecting variables if the last step of the simulation has completed
        # should be +1 but since we pass the step function it's actually +2
        if self.schedule.time == input_file.time_periods + 2:
            self.collect_vars()

    # does something when the last scientist in the TP is done investing in ideas
    def call_back(self):
        self.process_winners()

        unpack_model_arrays_data(self)

        # updating model dataframe
        df_model = pd.read_pickle(self.directory+'model_vars_df.pkl')
        df_model.loc[self.schedule.time] = [self.total_effort, self.effort_invested_by_age]
        df_model.to_pickle(self.directory+'model_vars_df.pkl')
        df_model = None

        store_model_arrays_data(self, False)

    def process_winners(self):
        investing_queue = pd.read_pickle('tmp/model/investing_queue.pkl')

        # initializing list of dictionaries with returns and investments for each idea in the TP
        list_dict = new_list_dict(self)

        # iterating through all investments by each scientists
        # young scientists get 0 returns, old scientists get all of the returns
        for index, row in investing_queue.iterrows():
            idx_idea = int(row['Idea Choice'])
            if list_dict[idx_idea]['Oldest ID'] < row['ID']:
                list_dict[idx_idea]['Oldest ID'] = row['ID']

            # update current stats for idea_dict
            list_dict[idx_idea] += Counter({"Total Effort": row['Marginal Effort'],
                                            "Total Increment": row['Increment'],
                                            "Max Return": row['Max Return'],
                                            "Actual Return": row['Actual Return']})

        # list that stores scientists who get returns (optional, see below)
        # happy_scientist = []

        # update model data collecting variables, and back for the old scientist who won all the returns for the idea
        for idea_choice in list_dict:
            idx_id = int(idea_choice["Oldest ID"]) - 1  # shift to left 1 since scientist id's start from 1
            if idx_id != -1:  # only active ideas needed
                # happy_scientist.append(idx_id)
                self.schedule.agents[idx_id].update(idea_choice)

        # updating data for remaining scientists who did not get any returns / inactive scientists
        # NOTE: OPTIONAL, only for agent_df. can speed up model simulation greatly if we don't care about
        # agent df with NaN for active scientists (happy_scientists would also be optional in this case)
        # for i in list(set(range(self.num_scientists))-set(happy_scientist)):  # difference between two sets
        #     self.schedule.agents[i].update(None)

        investing_queue = None
        list_dict = None
        idea_dict = None
        idea = None

    # for data collecting after model has finished running
    def collect_vars(self):
        unpack_model_arrays_data(self)
        idea = range(0, self.total_ideas, 1)
        tp = np.arange(self.total_ideas) // input_file.ideas_per_time
        prop_invested = self.total_effort / (2*input_file.true_means_lam)
        avg_k = np.round(divide_0(self.total_k, self.total_scientists_invested), 2)
        total_perceived_returns = np.round(self.total_perceived_returns, 2)
        total_actual_returns = np.round(self.total_actual_returns, 2)

        data1_dict = {"idea": idea,
                      "TP": tp,
                      "scientists_invested": self.total_scientists_invested,
                      "times_invested": self.total_times_invested,
                      "avg_k": avg_k,
                      "total_effort (marginal)": self.total_effort,
                      "prop_invested": prop_invested,
                      "total_pr": total_perceived_returns,
                      "total_ar": total_actual_returns}
        pd.DataFrame.from_dict(data1_dict).to_pickle(self.directory+'data1.pkl')
        store_model_arrays_data(self, False)

        unpack_model_lists(self)
        final_perceived_returns_invested_ideas_flat = flatten_list(self.final_perceived_returns_invested_ideas)
        ind_vars_dict = {"agent_k_invested_ideas": self.final_k_invested_ideas,
                         "agent_perceived_return_invested_ideas": final_perceived_returns_invested_ideas_flat,
                         "agent_actual_return_invested_ideas": self.final_actual_returns_invested_ideas}
        pd.DataFrame.from_dict(ind_vars_dict).to_pickle(self.directory+'ind_vars.pkl')
        store_model_lists(self, False)

        del final_perceived_returns_invested_ideas_flat, ind_vars_dict, data1_dict, idea, tp, prop_invested, avg_k,\
            total_perceived_returns, total_actual_returns


def mp_helper(model, i):
    model.schedule.agents[i].step()


# def scientist_creator(i, model):
#     return Scientist(i, model)
