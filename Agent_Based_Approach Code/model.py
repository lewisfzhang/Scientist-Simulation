from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np
from numpy.random import poisson
from functions import *  # anything not directly tied to Mesa objects
from optimize1 import *
import math


# This file has 3 parts
# 1. Agent class from Mesa
# 2. Model class from Mesa
class Scientist(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        # Scalar: amount of effort a scientist starts with in each time period
        # he or she is alive (not accounting for start effort decay)
        self.start_effort = poisson(lam=model.start_effort_lam)

        # Scalar: rate of decay for start_effort of old scientists; currently
        # set at 1 but can be adjusted as necessary
        # self.start_effort_decay = model.start_effort_decay

        # Scalar: amount of effort a scientist has left; goes down within a
        # given time period as a scientist invests in various ideas
        self.avail_effort = self.start_effort

        # Array: investment cost for each idea for a given scientist; a scientist
        # must first pay an idea's investment cost before receiving returns from
        # additional investment
        # (so each scientist has different cost for each idea)
        self.k = poisson(lam=model.k_lam, size=model.total_ideas)

        # error each scientist has for perceived compared to the true actual returns
        self.noise = model.noise_factor * np.random.normal(0,1,model.total_ideas)

        # Arrays: parameters determining perceived returns for ideas, which are
        # distinct from true returns. Ideas are modeled as logistic CDFs ("S" curve)
        self.sds = model.true_sds + self.noise
        self.means = model.true_means + self.noise

        # Create the ideas/returns matrix
        self.perceived_returns_matrix = create_return_matrix(model.total_ideas, self.sds, self.means, self.model.M,
                                                             model.true_sds_lam, model.true_means_lam)

        # Each scientist is assigned a unique ID. We also use unique IDs to determine
        # scientists' ages and thus which scientists are alive in a given time period
        # and which ideas they can invest in
        self.birth_order = math.ceil(2*unique_id/model.N)

        # Array: keeps track of how much effort a scientist has invested in each idea
        # NOTE: does NOT include effort paid for ideas' investment costs
        self.effort_invested_by_scientist = np.zeros(model.total_ideas)
        
        # Array: keeps track of which ideas can be worked on in a given time period
        # for a given scientist's age. This array will have 1s for ideas that
        # are accessible to a scientist and 0s for all other ideas
        self.avail_ideas = np.zeros(model.total_ideas)
        
        # Array: keeps track of total, perceived returns for each idea given
        # a scientist's level of investment in that idea
        self.perceived_returns = np.zeros(model.total_ideas)
        self.actual_returns = np.zeros(model.total_ideas)
        # Array: creates a copy of the max investment array from model class;
        # this contains the max amount of effort that can be invested in each
        # idea across all scientists
        # NOTE: this doesn't include effort paid to meet investment costs
        # self.max_investment = model.max_investment.copy()
        
        # Array: keeps track of effort invested ONLY during the current time period
        # NOTE: resets to 0 after each time period and DOES include investment costs
        self.eff_inv_in_period_increment = np.zeros(model.total_ideas)

        self.eff_inv_in_period_marginal = np.zeros(model.total_ideas)

        # Allows scientists to access model variables
        self.model = model

        # Array: keeping track of all the returns of investing in each available and invested ideas
        # NOTE: the K is based on initial learning cost, not current cost
        # self.final_perceived_returns_avail_ideas = []
        # self.final_k_avail_ideas = []
        # self.final_actual_returns_avail_ideas = []
        self.final_perceived_returns_invested_ideas = []
        self.final_k_invested_ideas = []
        self.final_actual_returns_invested_ideas = []

        self.update()

    def update(self):
        # conversions to tuple so dataframe updates (not sure why this happens with numpy arrays)
        self.model.total_effort_tuple = tuple(self.model.total_effort)
        self.effort_invested_by_scientist_tuple = tuple(self.effort_invested_by_scientist)
        self.eff_inv_in_period_increment_tuple = tuple(self.eff_inv_in_period_increment)
        self.eff_inv_in_period_marginal_tuple = tuple(self.eff_inv_in_period_marginal)
        self.perceived_returns_tuple = rounded_tuple(self.perceived_returns)
        self.actual_returns_tuple = rounded_tuple(self.actual_returns)
        # self.final_k_avail_ideas_tuple = tuple(self.final_k_avail_ideas)
        # self.final_perceived_returns_avail_ideas_tuple = rounded_tuple(self.final_perceived_returns_avail_ideas)
        # self.final_actual_returns_avail_ideas_tuple = rounded_tuple(self.final_actual_returns_avail_ideas)
        self.final_perceived_returns_invested_ideas_tuple = rounded_tuple(self.final_perceived_returns_invested_ideas)
        self.final_k_invested_ideas_tuple = rounded_tuple(self.final_k_invested_ideas)
        self.final_actual_returns_invested_ideas_tuple = rounded_tuple(self.final_actual_returns_invested_ideas)

        # list of numpy into tuple of tuple
        temp_list = []
        for i in range(len(self.model.effort_invested_by_age)):
            temp_list.append(tuple(self.model.effort_invested_by_age[i]))
        self.model.effort_invested_by_age_tuple = tuple(temp_list)

    def step(self):
        # Check a scientist's age in the current time period
        # NOTE: model time begins at 0
        self.current_age = (self.model.schedule.time - self.birth_order)

        # scientist is alive
        if 0 <= self.current_age < self.model.time_periods_alive and self.model.schedule.time >= 2:
            # Determine start/available effort based on the rate of decay for old scientists
            # Currently, start_effort_decay is set to 1, meaning there is no decay for
            # old scientists
            #
            # first old scientist didn't invest when he was young (CAN CHANGE IF NECESSARY)
            # remember that decay * current age shouldn't be greater than start_effort!!!
            #
            # not implementing decay until later...
            # if self.model.schedule.time != 2 and self.current_age > 0:
            #     self.start_effort = self.start_effort - self.start_effort_decay * self.current_age
            #     self.avail_effort = self.start_effort

            # reset effort in new time period

            # reset effort in new time period
            self.eff_inv_in_period_increment[:] = 0
            self.eff_inv_in_period_marginal[:] = 0

            # Array: has length equal to the total number of ideas in the model,
            # with 0s, 1s, 2s, etc. that indicate which time periods ideas are from
            idea_periods = np.arange(self.model.total_ideas) // self.model.ideas_per_time

            # reset effort
            self.avail_effort = self.start_effort

            # unlimited access to past ideas, too lazy to think of another way to implement double negative
            # what this statement really wants is idea_periods <= schedule.time
            self.avail_ideas = np.logical_not(idea_periods > self.model.schedule.time)
            greedy_investing(self)

        elif self.current_age == self.model.time_periods_alive:
            # reset effort in new time period
            self.eff_inv_in_period_increment[:] = 0
            self.eff_inv_in_period_marginal[:] = 0

            # reset returns because scientists is not active
            self.perceived_returns[:] = 0
            self.actual_returns[:] = 0

        self.model.steps_taken += 1
        if self.model.steps_taken == self.model.num_scientists:
            self.model.steps_taken = 0
            self.update()
            if self.model.schedule.time == self.model.time_periods+1:
                self.model.collect_vars()


class ScientistModel(Model):
    def __init__(self, time_periods, ideas_per_time, N, max_investment_lam, true_sds_lam, true_means_lam,  # ScientistModel variables
                 start_effort_lam, start_effort_decay, noise_factor, k_lam, sds_lam, means_lam, time_periods_alive,  #AgentModel variables
                 seed=None):

        super().__init__(seed)

        # for batch runs
        self.running = True

        # store variables into Scientist(Agent) objects
        self.start_effort_lam = start_effort_lam
        self.noise_factor = noise_factor
        self.k_lam = k_lam
        self.sds_lam = sds_lam
        self.means_lam = means_lam
        self.time_periods_alive = time_periods_alive
        self.time_periods = time_periods
        self.N = N

        # Scalar: indicates the total number of scientists in the model
        # N is the number of scientists per time period
        self.num_scientists = int(N/2)*(time_periods + 1)

        # Scalar: number of ideas unique to each time period
        self.ideas_per_time = ideas_per_time
        
        # Scalar: total number of ideas in the model. +2 is used to account
        # for first two, non-steady state time periods
        self.total_ideas = ideas_per_time*(time_periods+2)

        # scalars that store means of the mean and sds for returns
        self.true_sds_lam = true_sds_lam
        self.true_means_lam = true_means_lam

        # Array: store parameters for true idea return distribution
        self.true_sds = poisson(lam=self.true_sds_lam, size=self.total_ideas)
        self.true_means = poisson(lam=self.true_means_lam, size=self.total_ideas)
        
        # Ensures that none of the standard devs are equal to 0, this is OKAY
        self.true_sds += 1 + self.noise_factor

        # M is a scalar that multiples based on each idea
        self.M = poisson(lam=10000, size=self.total_ideas)

        # creates actual returns matrix
        self.actual_returns_matrix = create_return_matrix(self.total_ideas, self.true_sds, self.true_means, self.M,
                                                          self.true_sds_lam, self.true_means_lam)

        # Array: keeps track of total effort allocated to each idea across all scientists
        self.total_effort = np.zeros(self.total_ideas)

        # format = [young,old]
        self.effort_invested_by_age = [np.zeros(self.total_ideas), np.zeros(self.total_ideas)]

        # data collector (useless) variables
        self.total_perceived_returns = np.zeros(self.total_ideas)
        self.total_actual_returns = np.zeros(self.total_ideas)
        self.total_k = np.zeros(self.total_ideas)
        self.total_times_invested = np.zeros(self.total_ideas)
        self.total_scientists_invested = np.zeros(self.total_ideas)

        # Make scientists choose ideas and allocate effort in a random order
        # for each step of the model (i.e. within a time period, the order
        # in which young and old scientists get to invest in ideas is random)
        self.schedule = RandomActivation(self)

        # counts number of times/steps
        self.count_time = 0
        self.steps_taken = 0

        # creates Agent objects
        for i in range(1, self.num_scientists + 1):
            a = Scientist(i, self)
            self.schedule.add(a)
        
        # Create data collector method for keeping track of variables over time
        self.datacollector = DataCollector(
            model_reporters={"Total Effort List": "total_effort_tuple",
                             "Total Effort By Age": "effort_invested_by_age_tuple"},
            agent_reporters={"TP Born": "birth_order",
                             "Total effort invested": "effort_invested_by_scientist_tuple",
                             "Effort invested in period (increment)": "eff_inv_in_period_increment_tuple",
                             "Effort invested in period (marginal)": "eff_inv_in_period_marginal_tuple",
                             "Perceived returns": "perceived_returns_tuple",
                             "Actual returns": "actual_returns_tuple"})

    def step(self):
        # once count_time exceeds total time periods, program does nothing to prevent OutofBounds
        if self.count_time < self.time_periods+2:
            # Call data collector to keep track of variables at each model step
            self.schedule.step()
            self.datacollector.collect(self)

        self.count_time += 1

    # for data collecting after model has finished running
    def collect_vars(self):
        self.avg_k = np.round(divide_0(self.total_k, self.total_scientists_invested), 2)
        # avg_pr = np.round(np.divide(model.total_perceived_returns, model.total_times_invested,
        #                    out=np.zeros_like(model.total_perceived_returns), where=model.total_times_invested!=0),2)
        # avg_ar = np.round(np.divide(model.total_actual_returns, model.total_times_invested,
        #                    out=np.zeros_like(model.total_actual_returns), where=model.total_times_invested != 0),2)

        self.total_perceived_returns = np.round(self.total_perceived_returns, 2)
        self.total_actual_returns = np.round(self.total_actual_returns, 2)
        self.prop_invested = self.total_effort / (2*self.true_means_lam)
