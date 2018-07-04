# model.py

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
import config
import pickle
from collections import Counter, defaultdict
from store import *
from functools import partial
import threading
from multiprocessing.pool import ThreadPool
import random
from functions import *


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
        np.random.seed(config.seed_array[unique_id][0])
        self.start_effort = 150  # poisson(lam=config.start_effort_lam)  # utility

        # Scalar: amount of effort a scientist has left; goes down within a
        # given time period as a scientist invests in various ideas
        self.avail_effort = self.start_effort  # utility

        # the specific agent id unique to him/herself
        self.unique_id = unique_id  # utility

        # SCALAR: Each scientist is assigned a unique ID. We also use unique IDs to determine
        # scientists' ages and thus which scientists are alive in a given time period and which ideas they can invest in
        self.birth_order = math.ceil(unique_id / config.N)  # utility

        # create specific tmp directory for agent
        self.directory = config.tmp_loc['agent'] + str(self.unique_id) + '/'
        create_directory(self.directory)

        random.seed(config.seed_array[unique_id][1])
        self.sds = random.choice([i for i in range(5, 15)])  # follows N

        random.seed(config.seed_array[unique_id][2])
        self.means = random.choice([i for i in range(50, 150)])  # follows num_TP

        # Array: investment cost for each idea for a given scientist; a scientist
        # must first pay an idea's investment cost before receiving returns from
        # additional investment (so each scientist has different cost for each idea)
        #
        # plucking from random since scientists will inherently be better/worse than others at learning
        np.random.seed(config.seed_array[unique_id][3])
        self.k = np.rint(self.model.k * (np.random.normal(self.means, self.sds, self.model.total_ideas)/100))  # utility

        # ARRAY: error each scientist has for perceived compared to the true actual returns
        random.seed(config.seed_array[unique_id][4])
        self.sds = random.choice([i for i in range(1, config.end_limit)])

        # ARRAY: the 'noise' or error based on each specific scientist
        np.random.seed(config.seed_array[unique_id][5])
        self.noise = np.random.normal(0, self.sds, model.total_ideas)  # void

        # Arrays: parameters determining perceived returns for ideas, which are
        # distinct from true returns. Ideas are modeled as logistic CDFs ("S" curve)
        self.sds = self.model.true_sds + self.noise  # void
        self.means = self.model.true_means + self.noise  # void

        # ARRAY: Create the ideas/returns matrix
        # NOTE: logistic_cdf is not ranodm, always generates same curve based on means and sds
        self.perceived_returns_matrix = np.asarray([self.model.M, self.sds, self.means])  # utility

        # dereferencing 'void' variables
        self.noise = None
        self.sds = None
        self.means = None

        # Array: keeps track of how much effort a scientist has invested in each idea
        # NOTE: does NOT include effort paid for ideas' investment costs
        self.marginal_invested_by_scientist = np.zeros(self.model.total_ideas)  # utility
        self.k_invested_by_scientist = np.zeros(self.model.total_ideas)  # utility

        # Array: keeps track of total, perceived returns for each idea given
        # a scientist's level of investment in that idea
        self.perceived_returns = np.zeros(self.model.total_ideas)  # utility
        self.actual_returns = np.zeros(self.model.total_ideas)  # utility

        # two new data collecting variables for the TP used in update() function
        self.perceived_returns_tp = np.zeros(self.model.total_ideas)
        self.actual_returns_tp = np.zeros(self.model.total_ideas)

        # Array: keeps track of effort invested ONLY during the current time period
        # NOTE: resets to 0 after each time period and DOES include investment costs
        self.eff_inv_in_period_k = np.zeros(self.model.total_ideas)  # utility
        self.eff_inv_in_period_marginal = np.zeros(self.model.total_ideas)  # utility

        store_agent_arrays(self)
        store_agent_arrays_data(self)
        store_agent_arrays_tp(self)

    def step(self, lock):
        # Check a scientist's age in the current time period
        # NOTE: model time begins at 0
        self.current_age = self.model.schedule.time - self.birth_order

        # scientist is alive
        if 0 <= self.current_age < config.time_periods_alive and self.model.schedule.time >= 2:
            # pull all variables out of stored files
            unpack_agent_arrays(self)
            unpack_agent_arrays_tp(self)

            # reset effort and returns in new time period
            self.eff_inv_in_period_k[:] = 0
            self.eff_inv_in_period_marginal[:] = 0
            self.perceived_returns_tp[:] = 0
            self.actual_returns_tp[:] = 0

            # reset effort
            self.avail_effort = self.start_effort

            unpack_model_arrays(self.model, lock[0])
            # Array: keeps track of which ideas can be worked on in a given time period
            # for a given scientist's age. This array will have 1s for ideas that
            # are accessible to a scientist and 0s for all other ideas
            self.avail_ideas = np.logical_not(self.model.idea_periods > self.model.schedule.time)
            store_model_arrays(self.model, False, lock[0])

            greedy_investing(self, lock[1:])  # all locks except the first

            del self.avail_ideas
            store_agent_arrays(self)
            store_agent_arrays_tp(self)
            gc_collect()

        elif self.current_age == config.time_periods_alive:
            unpack_agent_arrays_data(self)
            unpack_agent_arrays_tp(self)

            # reset effort in new time period
            self.eff_inv_in_period_k[:] = 0
            self.eff_inv_in_period_marginal[:] = 0

            # reset returns because scientists is not active
            self.perceived_returns[:] = 0
            self.actual_returns[:] = 0

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
            self.perceived_returns_tp[idea] += df_row['Max Return']
            self.actual_returns[idea] += df_row['Actual Return']
            self.actual_returns_tp[idea] += df_row['Actual Return']
            del idea, df_row
        self.update_agent_df()
        store_agent_arrays_data(self)
        store_agent_arrays_tp(self)

    def update_agent_df(self):
        # updating agent dataframe
        df_agent = pd.read_pickle('tmp/model/agent_vars_df.pkl')
        # format: TP Born, Total effort invested, Effort invested in period (increment),
        #         Effort invested in period (marginal), Perceived returns, Actual returns
        new_data = {'TP Born': self.birth_order,
                    'Effort Invested In Period (K)': df_formatter(self.eff_inv_in_period_k, "effort"),
                    'Effort Invested In Period (Marginal)': df_formatter(self.eff_inv_in_period_marginal, "effort"),
                    'Perceived Returns': df_formatter(self.perceived_returns_tp, "returns"),
                    'Actual Returns': df_formatter(self.actual_returns_tp, "returns")}
        # s = df_formatter(self.perceived_returns_tp, "returns")
        # if s != '-':
        #     print(s)
        df_agent.loc[self.model.schedule.time].loc[self.unique_id] = new_data
        df_agent.to_pickle(self.model.directory+'agent_vars_df.pkl')

        # Dereferencing variables
        del df_agent, new_data


class ScientistModel(Model):
    def __init__(self, seed=None):
        super().__init__(seed)

        # create specific tmp directory for model
        self.directory = config.tmp_loc['model']
        create_directory(self.directory)

        # Scalar: indicates the total number of scientists in the model
        self.num_scientists = config.num_scientists  # utility

        # Scalar: total number of ideas in the model. +2 is used to account
        # for first two, non-steady state time periods
        self.total_ideas = config.ideas_per_time * (config.time_periods + 2)  # utility

        # Array: has length equal to the total number of ideas in the model,
        # with 0s, 1s, 2s, etc. that indicate which time periods ideas are from
        self.idea_periods = np.arange(self.total_ideas) // config.ideas_per_time  # utility

        # k is the learning cost for each idea
        np.random.seed(config.seed_array[0][0])
        self.k = poisson(lam=config.k_lam, size=self.total_ideas)  # void/utility (used for init agent objects)

        # Array: store parameters for true idea return distribution
        np.random.seed(config.seed_array[0][1])
        self.true_sds = poisson(lam=config.true_sds_lam, size=self.total_ideas)  # void
        np.random.seed(config.seed_array[0][2])
        self.true_means = poisson(lam=config.true_means_lam, size=self.total_ideas)  # void

        # Ensures that none of the standard devs are equal to 0, this is OKAY
        self.true_sds += 1  # void

        # M is a scalar that multiples based on each idea
        # not sure if this is redundant since we already have random poisson values for true_means and true_sds
        np.random.seed(config.seed_array[0][3])
        self.M = 100 * poisson(lam=100, size=self.total_ideas)  # void

        # creates actual returns matrix
        self.actual_returns_matrix = np.asarray([self.M, self.true_sds, self.true_means])  # utility

        # Array: keeps track of total effort allocated to each idea across all scientists
        self.total_effort = np.zeros(self.total_ideas)

        # data collector variable
        # format = [young,old]
        self.effort_invested_by_age = np.asarray([np.zeros(self.total_ideas), np.zeros(self.total_ideas)])

        # data collector variables that have no effect on results of model
        self.total_perceived_returns = np.zeros(self.total_ideas)
        self.total_actual_returns = np.zeros(self.total_ideas)
        self.total_k = np.zeros(self.total_ideas)
        self.total_times_invested = np.zeros(self.total_ideas)
        self.total_scientists_invested = np.zeros(self.total_ideas)
        self.total_scientists_invested_helper = [set() for i in range(self.total_ideas)]

        # Array: keeping track of all the returns of investing in each available and invested ideas
        # NOTE: the K is based on initial learning cost, not current cost
        self.final_perceived_returns_invested_ideas = [[] for i in range(self.num_scientists)]
        self.final_k_invested_ideas = [[] for i in range(self.num_scientists)]
        self.final_actual_returns_invested_ideas = [[] for i in range(self.num_scientists)]
        self.final_idea_idx = [[] for i in range(self.num_scientists)]
        self.final_scientist_id = [[] for i in range(self.num_scientists)]
        self.final_marginal_invested_ideas = [[] for i in range(self.num_scientists)]

        # Make scientists choose ideas and allocate effort in a random order
        # for each step of the model (i.e. within a time period, the order
        # in which young and old scientists get to invest in ideas is random)
        self.schedule = BaseScheduler(self)  # NOTE: doesn't skew results if not random due to call_back
        self.agent_dict = defaultdict(int)

        # creates Agent objects and adds them to the schedule
        # NOTE: with multithreading results will vary each time
        if config.use_multithreading:
            p = ThreadPool()
            m = mp.Manager()
            func = partial(create_scientists, m.Lock())
            p.starmap(func, [(self, i) for i in range(1, self.num_scientists + 1)])
            p.close()
            p.join()
        else:
            for i in range(1, self.num_scientists + 1):
                self.schedule.add(Scientist(i, self))
                self.agent_dict[i] = i-1  # shift index one to the left

        # dereferencing variables
        del self.k, self.true_sds, self.true_means, self.M

        create_datacollectors(self)
        create_list_dict(self)
        store_model_arrays(self, True, None)
        store_model_arrays_data(self, True, None)
        store_model_lists(self, True, None)

    def step(self):
        # queue format: idea_choice, scientist.marginal_effort[idea_choice], increment, max_return,
        #               actual_return, scientist.unique_id
        pd.DataFrame(columns=['Idea Choice', 'Max Return', 'ID']).to_pickle('tmp/model/investing_queue.pkl')
        self.total_effort_start = np.load(self.directory + 'total_effort.npy')

        # iterates through all scientists in the model
        # below is the same as self.schedule.step()
        if config.use_multiprocessing:
            # NOTE: Starting a process using this method is rather slow compared to using fork or forkserver.
            p = mp.Pool(processes=config.num_processors)
            m = mp.Manager()
            lock1 = m.Lock()  # for model_arrays
            lock2 = m.Lock()  # for model_arrays_data
            lock3 = m.Lock()  # for model_lists
            lock4 = m.Lock()  # for model investing queue df
            func = partial(mp_helper_spawn, [lock1, lock2, lock3, lock4])
            # split scientists by num_processes available
            agent_list = list(chunks(range(1, self.num_scientists+1), config.num_processors))
            p.starmap(func, [(self, i) for i in agent_list])
            p.close()
            p.join()
            del p, m, lock1, lock2, lock3, lock4, func, agent_list
        else:
            for i in range(self.num_scientists):
                # NoneType for locks is handled in store.py
                self.schedule.agents[i].step([None, None, None, None])

        self.call_back()
        self.schedule.time += 1
        del self.total_effort_start

        gc_collect()

        # run data collecting variables if the last step of the simulation has completed
        # should be +1 but since we pass the step function it's actually +2
        if self.schedule.time == config.time_periods + 2:
            self.collect_vars()

    # does something when the last scientist in the TP is done investing in ideas
    def call_back(self):
        unpack_model_arrays_data(self, None)
        self.process_winners()

        # updating model dataframe
        df_model = pd.read_pickle(self.directory+'model_vars_df.pkl')
        df_model.loc[self.schedule.time] = [df_formatter(self.total_effort, "effort"),
                                            "Young:\r\n" + df_formatter(self.effort_invested_by_age[0], "effort")
                                            + "\r\nOld:\r\n" + df_formatter(self.effort_invested_by_age[0], "effort")]
        df_model.to_pickle(self.directory+'model_vars_df.pkl')
        del df_model
        store_model_arrays_data(self, False, None)

    def process_winners(self):
        investing_queue = pd.read_pickle('tmp/model/investing_queue.pkl')

        # initializing list of dictionaries with returns and investments for each idea in the TP
        list_dict = new_list_dict(self)

        # iterating through all investments by each scientists
        # young scientists get 0 returns, old scientists get all of the returns
        for index, row in investing_queue.iterrows():
            idx_idea = int(row['Idea Choice'])
            # NOTE: OLDER SCIENTIST HAS THE YOUNGER UNIQUE ID!!!
            if list_dict[idx_idea]['Oldest ID'] > row['ID']:
                list_dict[idx_idea]['Oldest ID'] = row['ID']

            # update current stats for list_dict
            if list_dict[idx_idea]["Updated"] == False:
                stop_index = int(self.total_effort[idx_idea])
                start_index = int(self.total_effort_start[idx_idea])

                self.actual_returns_matrix = np.load(self.directory + 'actual_returns_matrix.npy')
                actual_return = get_returns(idx_idea, self.actual_returns_matrix, start_index, stop_index)

                list_dict[idx_idea] += Counter({"Actual Return": actual_return})
                self.total_actual_returns[idx_idea] += actual_return

                list_dict[idx_idea]["Updated"] = True  # so it never runs again
                del stop_index, start_index, self.actual_returns_matrix

            # max (perceived) return is not as accurate since scientists don't know
            # what others are investing --> only can account for in actual return
            list_dict[idx_idea] += Counter({"Max Return": row['Max Return']})
            del index, row, idx_idea

        # set that stores scientists who get returns (optional, see below)
        happy_scientist = set()

        # update model data collecting variables, and back for the old scientist who won all the returns for the idea
        for idx, idea_choice in enumerate(list_dict):
            idx_id = int(idea_choice["Oldest ID"])
            if idx_id != self.num_scientists + 1:  # only active ideas needed
                happy_scientist.add(idx_id)
                # print('\nid',idx_id,'idea',idea_choice['Idea Choice'])
                self.schedule.agents[self.agent_dict[idx_id]].update(idea_choice)
            del idx, idea_choice, idx_id

        # updating data for remaining scientists who did not get any returns / inactive scientists
        # NOTE: OPTIONAL, only for agent_df. can speed up model simulation greatly if we don't care about
        # agent df with NaN for active scientists (happy_scientists would also be optional in this case)
        # but then you wouldn't be able to see how much all scientists invested in this period
        if self.schedule.time >= 2:
            if config.all_scientists:
                range_list = range(1, self.num_scientists+1)
            else:
                interval = self.schedule.time - config.time_periods_alive
                if interval < 0:
                    interval = 0
                range_list = range(1 + interval * config.N, self.schedule.time * config.N + 1)
                del interval
            for i in list(set(range_list)-happy_scientist):  # difference between two sets
                self.schedule.agents[self.agent_dict[i]].update(None)
            del range_list

        del investing_queue, list_dict, happy_scientist

    # for data collecting after model has finished running
    def collect_vars(self):
        print("\n\n\ndone with step 9")
        start = timeit.default_timer()

        # PART 1
        unpack_model_arrays_data(self, None)
        idea = range(0, self.total_ideas, 1)
        tp = np.arange(self.total_ideas) // config.ideas_per_time
        prop_invested = rounded_tuple(self.total_effort / (2 * config.true_means_lam))
        avg_k = np.round(divide_0(self.total_k, self.total_scientists_invested), 2)
        total_perceived_returns = np.round(self.total_perceived_returns, 2)
        total_actual_returns = np.round(self.total_actual_returns, 2)

        ideas_dict = {"idea": idea,
                      "TP": tp,
                      "scientists_invested": self.total_scientists_invested,
                      "times_invested": self.total_times_invested,
                      "avg_k": avg_k,
                      "total_effort (marginal)": rounded_tuple(self.total_effort),
                      "prop_invested": prop_invested,
                      "total_pr": total_perceived_returns,
                      "total_ar": total_actual_returns}
        pd.DataFrame.from_dict(ideas_dict).replace(np.nan, '', regex=True).to_pickle(self.directory+'ideas.pkl')
        store_model_arrays_data(self, False, None)
        del ideas_dict, idea, tp, prop_invested, avg_k, total_perceived_returns, total_actual_returns


        # PART 2
        unpack_model_lists(self, None)
        ind_ideas_dict = {"idea_idx": rounded_tuple(flatten_list(self.final_idea_idx)),
                          "scientist_id": rounded_tuple(flatten_list(self.final_scientist_id)),
                          "agent_k_invested_ideas": rounded_tuple(flatten_list(self.final_k_invested_ideas)),
                          "agent_marginal_invested_ideas": rounded_tuple(flatten_list(self.final_marginal_invested_ideas)),
                          "agent_perceived_return_invested_ideas": rounded_tuple(flatten_list(self.final_perceived_returns_invested_ideas)),
                          "agent_actual_return_invested_ideas": rounded_tuple(flatten_list(self.final_actual_returns_invested_ideas))}
        pd.DataFrame.from_dict(ind_ideas_dict).to_pickle(self.directory+'ind_ideas.pkl')
        store_model_lists(self, False, None)
        del ind_ideas_dict


        # PART 3
        ind_vars = pd.read_pickle(self.directory + 'ind_ideas.pkl')
        agent_vars = pd.read_pickle(self.directory + 'agent_vars_df.pkl')

        actual_returns = agent_vars[agent_vars['Actual Returns'].str.startswith("{", na=False)]['Actual Returns']

        num_scientists = config.N * (config.time_periods + 1)  # same as in the model
        # format of scientist_tracker: [agent_id][num_ideas_invested][total_returns]
        returns_tracker = np.zeros(num_scientists)

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
        np.save(self.directory+'social_output.npy', np.asarray(y_var))
        np.save(self.directory+'ideas_entered.npy', np.asarray(x_var))
        del ind_vars, agent_vars, actual_returns, num_scientists, returns_tracker, curr_id, counter_x, x_var, y_var


        # PART 4
        agent_vars = pd.read_pickle(self.directory + 'agent_vars_df.pkl').replace(np.nan, '', regex=True)
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
        np.save(self.directory+'prop_age.npy', prop_age)
        del prop_age, age_tracker, agent_vars


        # PART 5
        agent_vars = pd.read_pickle(self.directory + 'agent_vars_df.pkl').replace(np.nan, '', regex=True)
        agent_marg = agent_vars[agent_vars['Effort Invested In Period (Marginal)'].str.startswith("{", na=False)]['Effort Invested In Period (Marginal)']
        self.idea_periods = np.load(self.directory + "idea_periods.npy")

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

                idea_age = idx[0] - self.idea_periods[idea]  # current tp - tp born
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
        np.save(self.directory + "marginal_effort_by_age.npy", marginal_effort)
        np.save(self.directory + "prop_idea.npy", prop_idea)
        del agent_vars, agent_marg, self.idea_periods, marginal_effort, prop_idea, total_ideas

        print("time elapsed:", timeit.default_timer()-start)


def mp_helper_spawn(lock, model, agent_list):
    for i in agent_list:
        model.schedule.agents[model.agent_dict[i]].step(lock)


def create_scientists(lock, model, i):
    a = Scientist(i, model)
    lock.acquire()
    model.agent_dict[i] = len(model.schedule.agents)  # index of the scientist in the schedule
    model.schedule.add(a)
    lock.release()
    del a
