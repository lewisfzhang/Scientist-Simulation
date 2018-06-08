from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np
from numpy.random import poisson
from functions import *  # anything not directly tied to Mesa objects
from optimize import *

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
        self.start_effort_decay = model.start_effort_decay

        # Scalar: amount of effort a scientist has left; goes down within a
        # given time period as a scientist invests in various ideas
        self.avail_effort = self.start_effort

        # Array: investment cost for each idea for a given scientist; a scientist
        # must first pay an idea's investment cost before receiving returns from
        # additional investment
        # (so each scientist has different cost for each idea)
        self.k = poisson(lam=model.k_lam, size=model.total_ideas)

        # Arrays: parameters determining perceived returns for ideas, which are
        # distinct from true returns. Ideas are modeled as logistic CDFs ("S" curve)
        self.sds = poisson(lam=model.sds_lam, size=model.total_ideas)
        self.means = poisson(lam=model.sds_lam, size=model.total_ideas)
        # Ensures that none of the standard devs are equal to 0
        # NOTE: May be worth asking Jay if there is a better way to handle this
        self.sds += 1

        # Create the ideas/returns matrix
        self.perceived_returns_matrix = create_return_matrix(model.total_ideas, max(model.max_investment),
                                                             self.sds, self.means)

        self.actual_returns_matrix = create_return_matrix(model.total_ideas, max(model.max_investment),
                                                          self.model.true_sds, self.model.true_means)
        # Each scientist is assigned a unique ID. We also use unique IDs to determine
        # scientists' ages and thus which scientists are alive in a given time period
        # and which ideas they can invest in
        self.birth_order = unique_id
        
        # Array: keeps track of how much effort a scientist has invested in each idea
        # NOTE: does NOT include effort paid for ideas' investment costs
        self.effort_invested = np.zeros(model.total_ideas)
        
        # Array: keeps track of which ideas can be worked on in a given time period
        # for a given scientist's age. This array will have 1s for ideas that
        # are accessible to a scientist and 0s for all other ideas
        self.avail_ideas = np.zeros(model.total_ideas)
        
        # Array: keeps track of total, perceived returns for each idea given
        # a scientist's level of investment in that idea
        self.perceived_returns = np.zeros(model.total_ideas)
        
        # Array: creates a copy of the max investment array from model class;
        # this contains the max amount of effort that can be invested in each
        # idea across all scientists
        # NOTE: this doesn't include effort paid to meet investment costs
        self.max_investment = model.max_investment.copy()
        
        # Array: keeps track of effort invested ONLY during the current time period
        # NOTE: resets to 0 after each time period and DOES include investment costs
        self.eff_inv_in_period = np.zeros(model.total_ideas)
        
        # Allows scientists to access model variables
        self.model = model

        # Array: keeping track of all the returns of investing in each available ideas
        self.final_returns_avail_ideas = np.array([])
        self.final_k_avail_ideas = np.array([])

    # code reusability: code for each age of scientist into a single function
    def investing(self):
        # Scientists continue to invest in ideas until they run out of
        # available effort, or there are no more ideas to invest in
        while self.avail_effort > 0:

            # Array: determine which ideas scientists have invested no effort into
            no_effort_inv = (self.effort_invested == 0)

            # Array: calculate current investment costs based on prior investments;
            # has investment costs or 0 if scientist has already paid cost
            self.curr_k = no_effort_inv * self.k

            # Array (size = model.total_ideas): how much more effort can
            # be invested in a given idea based on the max investment for
            # that idea and how much all scientists have already invested
            self.effort_left_in_idea = self.max_investment - self.model.total_effort

            # Change current investment cost to 0 if a given idea has 0
            # effort left in it. This prevents the idea from affecting
            # the marginal effort
            for idx, value in enumerate(self.effort_left_in_idea):
                if value == 0:
                    self.curr_k[idx] = 0

            # Scalar: want to pull returns for expending self.increment units
            # of effort, where increment equals the max invest cost across
            # all ideas that haven't been invested in yet plus 1
            self.increment = max(self.curr_k[self.avail_ideas]) + 1

            # Edge case in which scientist doesn't have enough effort to invest
            # in idea with greatest investment cost
            # POTENTIAL BIAS! LESS EFFORT SHOULD CORRESPOND TO LESS RESULTS?!
            if self.avail_effort < self.increment:
                self.increment = self.avail_effort

            # Array: contains equivalent "marginal" efforts across ideas
            # For idea with the max invest cost, marginal_effort will equal 1
            # All others - marginal_effort is equal to increment minus invest cost, if any
            self.marginal_effort = self.increment - self.curr_k

            # Selects idea that gives the max return given equivalent "marginal" efforts
            # NOTE: See above for comments on calc_cum_returns function,
            # and exceptions on when the idea with the max return isn't chosen
            self.idea_choice, self.max_return = calc_cum_returns(self, self.model)

            # Accounts for the edge case in which max_return = 0 (implying that a
            # scientist either can't invest in ANY ideas [due to investment
            # cost barriers or an idea reaching max investment]). Effort
            # is lost and doesn't carry over to the next period
            if self.max_return == 0:
                print("i ran")
                self.avail_effort = 0
                continue

            # Accounts for the edge case in which scientist chooses to invest in
            # an idea that has less effort remaining than a scientist's
            # marginal effort
            if self.marginal_effort[self.idea_choice] > self.effort_left_in_idea[self.idea_choice]:
                self.marginal_effort[self.idea_choice] = self.effort_left_in_idea[self.idea_choice]
                self.increment = self.curr_k[self.idea_choice] + self.marginal_effort[self.idea_choice]

            # Updates parameters after idea selection and effort expenditure
            # NOTE: self.avail_effort and self.eff_inv_in_period should be
            # updated by the increment, not by marginal effort, because the
            # increment includes investment costs. We don't care about
            # paid investment costs for the other variables
            self.model.total_effort[self.idea_choice] += self.marginal_effort[self.idea_choice]
            self.effort_invested[self.idea_choice] += self.marginal_effort[self.idea_choice]
            self.eff_inv_in_period[self.idea_choice] += self.increment
            self.avail_effort -= self.increment
            self.perceived_returns[self.idea_choice] += self.max_return  # constant in 2-period lifespan scientists

        # system debugging print statements
        print("\ncurrent age", self.current_age, "   id", self.unique_id, "    step", self.model.schedule.time,
              "\navail ideas array:",self.avail_ideas.tolist(),
              "\ntotal effort invested array:", self.effort_invested.tolist(),
              "\neffort invested in time period:", self.eff_inv_in_period.tolist())

    def step(self):
        # Check a scientist's age in the current time period
        # NOTE: model time begins at 0 --> WRONG, begins at 1
        # NOTE: +3 ensures that we correctly calculate the current age; see
        # notes on time periods and ages in documentation
        self.current_age = (self.model.schedule.time - self.birth_order)
        # Array: has length equal to the total number of ideas in the model,
        # with 0s, 1s, 2s, etc. that indicate which time periods ideas are
        # from
        idea_periods = np.arange(self.model.total_ideas) // self.model.ideas_per_time

        # # scientists have died
        # if self.current_age >= 2:
        #     # Scientists are alive and able to do things for only two time periods
        #     # (i.e. can't do anything when they are not 0 or 1)
        #     # RESET EVERYTHING TO 0, since they are inactive like an unborn scientist
        #     self.eff_inv_in_period[:] = 0
        #     self.perceived_returns[:] = 0

        # Young scientist (added the AND condition to ensure in TP 1 the scientist doesn't do anything)
        if self.current_age == 0 and self.model.schedule.time >= 2:
            # Array: Contains 1s for ideas in the current time period and previous
            # time period, indicating that a young scientist can invest in those
            # ideas. The rest of the array has 0s
            self.avail_ideas = np.logical_or(idea_periods == self.model.schedule.time,
                idea_periods == (self.model.schedule.time - 1))
            self.investing()

        # Old scientist
        elif self.current_age == 1:
            # Array: Contains 1s for ideas in the current time period and previous
            # two time periods, indicating that an old scientist can invest in those
            # ideas. The rest of the array has 0s
            self.avail_ideas = np.logical_or(np.logical_or(idea_periods == self.model.schedule.time,
                idea_periods == (self.model.schedule.time - 1)), idea_periods == (self.model.schedule.time - 2))

            # Determine start/available effort based on the rate of decay for old scientists
            # Currently, start_effort_decay is set to 1, meaning there is no decay for
            # old scientists
            #
            # first old scientist didn't invest when he was young (CAN CHANGE IF NECESSARY)
            # remember that decay * current age shouldn't be greater than start_effort!!!
            if self.model.schedule.time != 2:
                self.start_effort = self.start_effort - self.start_effort_decay * self.current_age
                self.avail_effort = self.start_effort
            # Reset effort invested in the current time period
            self.eff_inv_in_period[:] = 0

            self.investing()

        # scientists not born yet
        else:
            self.eff_inv_in_period[:] = 0
            self.perceived_returns[:] = 0

        # conversions to tuple so dataframe updates (not sure why this happens with numpy arrays)
        self.model.total_effort_tuple = tuple(self.model.total_effort)
        self.effort_invested_tuple = tuple(self.effort_invested)
        self.eff_inv_in_period_tuple = tuple(self.eff_inv_in_period)
        self.perceived_returns_tuple = tuple(map(lambda x: isinstance(x, float) and round(x,5) or x, tuple(self.perceived_returns)))
        self.final_k_avail_ideas_tuple = tuple(self.final_k_avail_ideas)
        self.final_returns_avail_ideas_tuple = tuple(map(lambda x: isinstance(x, float) and round(x,5) or x, tuple(self.final_returns_avail_ideas)))

class ScientistModel(Model):
    def __init__(self, time_periods, ideas_per_time, N, max_investment_lam, true_sds_lam, true_means_lam,  # ScientistModel variables
                 start_effort_lam, start_effort_decay, k_lam, sds_lam, means_lam, #AgentModel variables
                 seed=None):

        super().__init__(seed)

        # for batch runs
        self.running = True

        # store variables into Scientist(Agent) objects
        self.start_effort_lam = start_effort_lam
        self.start_effort_decay = start_effort_decay
        self.k_lam = k_lam
        self.sds_lam = sds_lam
        self.means_lam = means_lam

        self.time_periods = time_periods
        self.N = N

        # Scalar: indicates the total number of scientists in the model
        # N is the number of scientists per time period
        self.num_scientists = time_periods + 1  # NOTE: ONLY WHEN N=2!!!

        # Scalar: number of ideas unique to each time period
        self.ideas_per_time = ideas_per_time
        
        # Scalar: total number of ideas in the model. +2 is used to account
        # for first two, non-steady state time periods
        self.total_ideas = ideas_per_time*(time_periods+2)

        self.max_investment_lam = max_investment_lam
        # Array: stores the max investment allowed for each idea
        self.max_investment = poisson(lam=max_investment_lam, size=self.total_ideas)

        # Array: store parameters for true idea return distribution
        self.true_sds = poisson(lam=true_sds_lam, size=self.total_ideas)
        self.true_means = poisson(lam=true_means_lam, size=self.total_ideas)
        
        # Ensures that none of the standard devs are equal to 0
        self.true_sds += 1
        
        # Array: keeps track of total effort allocated to each idea across all
        # scientists
        self.total_effort = np.zeros(self.total_ideas)

        # Make scientists choose ideas and allocate effort in a random order
        # for each step of the model (i.e. within a time period, the order
        # in which young and old scientists get to invest in ideas is random)
        self.schedule = RandomActivation(self)

        # counts number of times/steps
        self.count_time = 0

        for i in range(1, self.num_scientists + 1):
            a = Scientist(i, self)
            self.schedule.add(a)
        
        # Create data collector method for keeping track of variables over time
        self.datacollector = DataCollector(
            model_reporters={"Total Effort Sum": get_total_effort, "Total Effort List": "total_effort_tuple"},
            agent_reporters={"Total effort invested": "effort_invested_tuple", "Effort invested in period": "eff_inv_in_period_tuple",
                             "Perceived returns": "perceived_returns_tuple"})

    def step(self):
        # once count_time exceeds total time periods, program does nothing to prevent OutofBounds
        if self.count_time < self.time_periods+2:
            # Call data collector to keep track of variables at each model step
            self.schedule.step()
            self.datacollector.collect(self)

        self.count_time += 1
