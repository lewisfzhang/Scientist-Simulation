# model.py

from mesa import Agent, Model
from mesa.time import BaseScheduler
from numpy.random import poisson
from optimize import *
import math
import multiprocessing as mp
from collections import defaultdict
from store import *
from functools import partial
from multiprocessing.pool import ThreadPool
import random
from functions import *
from final import *
from process import *


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
        # can set to constant if needed rather than poisson distribution
        np.random.seed(config.seed_array[unique_id][0])
        self.start_effort = poisson(lam=config.start_effort_lam)  # utility

        # Scalar: amount of effort a scientist has left; goes down within a
        # given time period as a scientist invests in various ideas
        self.avail_effort = self.start_effort  # utility

        # the specific agent id unique to him/herself
        self.unique_id = unique_id  # utility

        # SCALAR: Each scientist is assigned a unique ID. We also use unique IDs to determine
        # scientists' ages and thus which scientists are alive in a given time period and which ideas they can invest in
        self.birth_order = math.ceil(unique_id / config.N)  # utility

        # create specific tmp directory for agent
        self.directory = config.tmp_loc + 'agent_' + str(self.unique_id) + '/'
        create_directory(self.directory)

        random.seed(config.seed_array[unique_id][1])
        self.sds = random.choice(np.arange(5, 15))  # follows N

        random.seed(config.seed_array[unique_id][2])
        self.means = random.choice(np.arange(50, 150))  # follows num_TP

        # Array: investment cost for each idea for a given scientist; a scientist
        # must first pay an idea's investment cost before receiving returns from
        # additional investment (so each scientist has different cost for each idea)
        #
        # plucking from random since scientists will inherently be better/worse than others at learning
        np.random.seed(config.seed_array[unique_id][3])
        self.k = np.rint(self.model.k * (np.random.normal(self.means, self.sds, self.model.total_ideas)/100))  # utility

        # Arrays: parameters determining perceived returns for ideas, which are
        # distinct from true returns. Ideas are modeled as logistic CDFs ("S" curve)
        self.M = self.model.M + random_noise(4, 5, self.unique_id, self.model.total_ideas, config.true_M)  # void
        self.sds = self.model.true_sds + random_noise(6, 7, self.unique_id, self.model.total_ideas,
                                                      config.true_sds_lam)  # void
        self.means = self.model.true_means + random_noise(8, 9, self.unique_id, self.model.total_ideas,
                                                          config.true_means_lam)  # void
        if config.use_idea_shift:
            self.idea_shift = self.model.true_means + random_noise(10, 11, self.unique_id, self.model.total_ideas,
                                                                   config.true_idea_shift)  # void
        else:
            self.idea_shift = np.zeros(self.model.total_ideas)

        # ARRAY: Create the ideas/returns matrix
        # NOTE: logistic_cdf is not ranodm, always generates same curve based on means and sds
        self.perceived_returns_matrix = np.asarray([self.M, self.sds, self.means, self.idea_shift])  # utility

        # check if anything is < 0 (that is a concern!)
        r, c = np.where(self.perceived_returns_matrix < 0)
        for i in range(len(r)):
            print("agent_id", self.unique_id, 'r', r[i], 'c', c[i])
            print(self.perceived_returns_matrix[r[i]][c[i]])
        del r, c

        # dereferencing 'void' variables
        self.noise = None
        self.sds = None
        self.means = None

        # Array: keeps track of how much effort a scientist has invested in each idea
        # NOTE: does NOT include effort paid for ideas' investment costs
        self.marginal_invested_by_scientist = np.zeros(self.model.total_ideas)  # utility
        self.k_invested_by_scientist = np.zeros(self.model.total_ideas)  # utility

        # two new data collecting variables for the TP used in update() function
        self.perceived_returns_tp = np.zeros(self.model.total_ideas)
        self.actual_returns_tp = np.zeros(self.model.total_ideas)

        # Array: keeps track of effort invested ONLY during the current time period
        # NOTE: resets to 0 after each time period and DOES include investment costs
        self.eff_inv_in_period_k = np.zeros(self.model.total_ideas)  # utility
        self.eff_inv_in_period_marginal = np.zeros(self.model.total_ideas)  # utility

        store_agent_arrays(self)
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

            investing_helper(self, lock[1:])  # all locks except the first

            del self.avail_ideas
            store_agent_arrays(self)
            store_agent_arrays_tp(self)

        elif self.current_age == config.time_periods_alive:
            unpack_agent_arrays_tp(self)

            # reset effort in new time period
            self.eff_inv_in_period_k[:] = 0
            self.eff_inv_in_period_marginal[:] = 0

            store_agent_arrays_tp(self)

    # converts mutable numpy arrays into easily accessed tuples
    def update(self, df_row):
        unpack_agent_arrays_tp(self)
        if df_row is not None:
            # Updates parameters after idea selection and effort expenditure
            # NOTE: self.avail_effort and self.eff_inv_in_period should be
            # updated by the increment, not by marginal effort, because the
            # increment includes investment costs. We don't care about
            # paid investment costs for the other variables
            idea = int(df_row['Idea Choice'])
            self.perceived_returns_tp[idea] += df_row['Max Return']
            self.actual_returns_tp[idea] += df_row['Actual Return']
            del idea, df_row
        # format: TP Born, Total effort invested, Effort invested in period (increment),
        #         Effort invested in period (marginal), Perceived returns, Actual returns
        new_data = {'TP Born': self.birth_order,
                    'Effort Invested In Period (K)': df_formatter(self.eff_inv_in_period_k, "effort"),
                    'Effort Invested In Period (Marginal)': df_formatter(self.eff_inv_in_period_marginal, "effort"),
                    'Perceived Returns': df_formatter(self.perceived_returns_tp, "returns"),
                    'Actual Returns': df_formatter(self.actual_returns_tp, "returns")}
        update_agent_df(self, new_data)
        store_agent_arrays_tp(self)
        del new_data


class ScientistModel(Model):
    def __init__(self, seed=None):
        super().__init__(seed)

        # create specific tmp directory for model
        self.directory = config.tmp_loc + 'model/'
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

        # ARRAY: store parameters for true idea return distribution
        np.random.seed(config.seed_array[0][1])
        self.true_sds = poisson(lam=config.true_sds_lam, size=self.total_ideas)  # void
        np.random.seed(config.seed_array[0][2])
        self.true_means = poisson(lam=config.true_means_lam, size=self.total_ideas)  # void

        # ARRAY: shifts the logistic cdf to the right
        # NOTE: not sure if this is necessary given updated logistic cdf formula
        if config.use_idea_shift:
            np.random.seed(config.seed_array[0][3])
            self.true_idea_shift = np.random.choice(np.arange(config.true_idea_shift), size=self.total_ideas)
        else:
            self.true_idea_shift = np.zeros(self.total_ideas)

        # Ensures that none of the standard devs are equal to 0, this is OKAY
        self.true_sds += 1  # void

        # M is a scalar that multiples based on each idea
        # not sure if this is redundant since we already have random poisson values for true_means and true_sds
        np.random.seed(config.seed_array[0][4])
        self.M = 100 * poisson(lam=config.true_M, size=self.total_ideas)  # void

        # creates actual returns matrix  # utility
        self.actual_returns_matrix = np.asarray([self.M, self.true_sds, self.true_means, self.true_idea_shift])

        # 0.001 * config.true_sds_lam / self.true_sds --> relative slope indicators
        # NOTE: not sure if this works properly since logistics are exponential, but derived based on intuition!
        self.idea_phase_label = logistic_cdf_inv_deriv(0.001 * config.true_sds_lam / self.true_sds, self.true_means, self.true_sds)

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
        self.total_idea_phase = np.zeros(3)

        # Array: keeping track of all the returns of investing in each available and invested ideas
        # NOTE: the K is based on initial learning cost, not current cost
        self.final_perceived_returns_invested_ideas = [[] for i in range(self.num_scientists)]
        self.final_k_invested_ideas = [[] for i in range(self.num_scientists)]
        self.final_actual_returns_invested_ideas = [[] for i in range(self.num_scientists)]
        self.final_idea_idx = [[] for i in range(self.num_scientists)]
        self.final_scientist_id = [[] for i in range(self.num_scientists)]
        self.final_marginal_invested_ideas = [[] for i in range(self.num_scientists)]
        self.final_slope = [[[] for i in range(self.num_scientists)], [[] for i in range(self.num_scientists)]]

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
        del self.k, self.true_sds, self.true_means, self.M, self.true_idea_shift

        create_datacollectors(self)
        create_list_dict(self)
        store_model_arrays(self, True, None)
        store_model_arrays_data(self, True, None)
        store_model_lists(self, True, None)
        store_actual_returns(self, None)

    def step(self):
        new_investing_queue(self)
        get_total_start_effort(self)

        # iterates through all scientists in the model
        # below is the same as self.schedule.step()
        if config.use_multiprocessing:
            # NOTE: Starting a process using this method is rather slow compared to using fork or forkserver.
            p = mp.Pool(processes=config.num_processors)
            m = mp.Manager()
            lock1 = m.Lock()  # for model_arrays
            lock2 = m.Lock()  # for model_arrays_data
            lock3 = m.Lock()  # for model_lists
            lock4 = m.Lock()  # for actual returns matrix
            lock5 = m.Lock()  # for model investing queue df
            func = partial(mp_helper_spawn, [lock1, lock2, lock3, lock4, lock5])
            # split scientists by num_processes available
            agent_list = list(chunks(range(1, self.num_scientists+1), config.num_processors))
            p.starmap(func, [(self, i) for i in agent_list])
            p.close()
            p.join()
            del p, m, lock1, lock2, lock3, lock4, func, agent_list
        else:
            for i in range(self.num_scientists):
                # NoneType for locks is handled in store.py
                self.schedule.agents[i].step([None, None, None, None, None])

        self.call_back()
        self.schedule.time += 1

        del self.total_effort_start
        gc_collect()

        # run data collecting variables if the last step of the simulation has completed
        # should be +1 but since we pass the step function it's actually +2
        if self.schedule.time == config.time_periods + 2:
            save_model(self)
            collect_vars(self)

    # does something when the last scientist in the TP is done investing in ideas
    def call_back(self):
        unpack_model_arrays_data(self, None)
        if config.use_equal:
            process_winners_equal(self)
        else:
            process_winners_old(self)

        # updating model dataframe
        data_list = [df_formatter(self.total_effort, "effort"),
                     "Young:\r\n" + df_formatter(self.effort_invested_by_age[0], "effort")
                     + "\r\nOld:\r\n" + df_formatter(self.effort_invested_by_age[0], "effort")]
        update_model_df(self, data_list)

        store_model_arrays_data(self, False, None)
        del data_list


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
