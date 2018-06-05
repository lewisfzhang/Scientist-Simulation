from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import numpy as np
from numpy.random import poisson, logistic
import sys

# Input: Parameters for the logistic cumulative distribution function
# Output: Value at x of the logistic cdf defined by the location and scale parameter
def logistic_cdf(x, loc, scale):
    return 1/(1+np.exp((loc-x)/scale))

# Input: 
# 1) num_ideas (scalar): number of ideas to create the return matrix for
# 2) max_of_max_inv (scalar): the maximum of all the maximum investment limits over all ideas
# 3) sds (array): the standard deviation parameters of the return curve of each idea
# 4) means (array): the mean parameters of the return curve of each idea
# Output:
# A matrix that is has dimension n x m where n = num_ideas and m = max_of_max_inv
# where each cell (i,j) contains the return based on the logistic cumulative 
# distribution function of the i-th idea and the j-th extra unit of effort
def create_return_matrix(num_ideas, max_of_max_inv, sds, means):

    # Creates array of the effort units to calculate returns for
    x = np.arange(max_of_max_inv + 1)
    returns_list = []
    for i in range(num_ideas):
        # Calculates the return for an idea for all amounts of effort units
        returns = logistic_cdf(x, loc = means[i], scale = sds[i])
        # Stacks arrays horizontally
        to_subt_temp = np.hstack((0,returns[:-1]))
        # Calculates return per unit of effort
        returns = returns - to_subt_temp
        returns_list.append(returns)
    return(np.array(returns_list))

# Input: 
# 1) numbers (array): contains the numbers we are picking the second largest from
# Output:
# The second largest number out of the array 
def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None


# Function to calculate "marginal" returns for available ideas, taking into account investment costs
# Input:
# 1) avail_ideas (array): indexes which ideas are available to a given scientist
# 2) total_effort (array):  contains cumulative effort already expended by all scientists
# 3) max_investment (array): contains the max possible investment for each idea
# 4) marginal_effort (array): "equivalent" marginal efforts, which equals
#    the max, current investment cost plus one minus individual investment costs for available ideas
#
# Output:
# 1) idx_max_return (scalar): the index of the idea the scientist chose to invest in
# 2) max_return (scalar): the perceived return of the associated, chosen idea
    
def calc_cum_returns(scientist, model):

    # Array: keeping track of all the returns of investing in each available ideas
    final_returns_avail_ideas = np.array([])

    # Scalar: limit on the amount of effort that a scientist can invest in a single idea
    # in one time period
    invest_cutoff = round(scientist.start_effort * 0.6)

    # Loops over all the ideas the scientist is allowed to invest in 
    for idea in np.where(scientist.avail_ideas == True)[0]:

        # OR Conditions
        # 1st) Edge case in which scientist doesn't have enough effort to invest in
        # an idea given the investment cost
        # 2nd) Ensures that effort invested in ideas doesn't go over max investment limit
        if scientist.marginal_effort[idea] <= 0 or scientist.effort_left_in_idea[idea] == 0:
            final_returns_avail_ideas = np.append(final_returns_avail_ideas, 0)

        # For instances in which a scientist's marginal effort exceeds the 
        # effort left in a given idea, calculate the returns for investing
        # exactly the effort left
        elif scientist.marginal_effort[idea] > scientist.effort_left_in_idea[idea]:
            start_index = int(model.total_effort[idea])
            stop_index = int(start_index + scientist.effort_left_in_idea[idea])
            returns = scientist.returns_matrix[idea, np.arange(start_index, stop_index)]
            total_return = sum(returns)
            final_returns_avail_ideas = np.append(final_returns_avail_ideas, total_return)

        # The case in which there are no restrictions for investing in this idea
        else:
            start_index = int(model.total_effort[idea])
            stop_index = int(start_index + scientist.marginal_effort[idea])
            returns = scientist.returns_matrix[idea, np.arange(start_index, stop_index)]
            total_return = sum(returns)
            final_returns_avail_ideas = np.append(final_returns_avail_ideas, total_return)


    # Finds the maximum return over all the available ideas
    max_return = max(final_returns_avail_ideas)

    # Finds the index of the maximum return over all the available ideas
    idx_max_return = np.where(final_returns_avail_ideas == max_return)[0]

    # Resolves edge case in which there are multiple max returns. In this case,
    # check if each idea associated with the max return has reached the
    # investment cutoff. If it has reached the cutoff, remove it from the array
    # of ideas with max returns. Then randomly choose among the remaining ideas

    # If there are ties for the maximum return
    if len(idx_max_return) > 1:

        # Array to keep track of updated index based on just the ideas that are tied for the max return
        upd_idx_max = np.array([])

        # Iterate over all indices of the array containing the indices of the tied max returns
        for idx in np.nditer(idx_max_return):

            # Deriving the correct index for the eff_inv_in_period array that contains all ideas that will
            # ever be created instead of only this scientist's available ideas that idx_max_return is in 
            # reference to
            idea_choice = idx + [(model.schedule.time + 1)*(model.ideas_per_time)] - len(final_returns_avail_ideas)

            # If the scientist has exceeded her own limit on investing in an idea, skip to the next idea
            if scientist.eff_inv_in_period[idea_choice] >= invest_cutoff:
                continue
            # Otherwise append it to the updated-index-max array as an eligible idea
            else:
                upd_idx_max = np.append(upd_idx_max, idea_choice)

        # Randomly choose an idea over all tied, eligible ideas
        idea_choice = int(np.random.choice(upd_idx_max))

        # Return both the idea_choice (this index is relative to the eff_inv_in_period array), and the max return
        return(idea_choice, max_return)

    # Otherwise if there are no ties
    else:

        # Deriving the correct index for the eff_inv_in_period array that contains all ideas that will
        # ever be created instead of only this scientist's available ideas that idx_max_return is in 
        # reference to
        idea_choice = idx_max_return[0] + [(model.schedule.time + 1)*(model.ideas_per_time)] - len(final_returns_avail_ideas)

        # Prevents scientists from investing further in a single idea if the
        # scientist has already invested at or above the investment cutoff.
        if scientist.eff_inv_in_period[idea_choice] >= invest_cutoff:
            second_max = second_largest(final_returns_avail_ideas)
            # second_max equal to 0 implies that the scientist can't invest
            # in any other ideas due to 1) ideas reaching max investment or 
            # 2) ideas having too high of an investment cost given the 
            # scientist's current available effort. In this edge case, the
            # scientist is allowed to continue investing in an idea past
            # the investment cutoff

            # The nested edge case where the scientist doesn't have any other ideas to invest in
            # due to other restrictions
            if second_max == 0:
                # Bypass the restriction on the investment cutoff in this case
                return idea_choice, max_return
            # If the scientist does have other ideas she can invest in 
            else:
                # Find the idea with the second highest return
                idx_second_max = np.where(final_returns_avail_ideas == second_max)[0]

                # If there are ties for the second highest return
                if len(idx_second_max) > 1:
                    # Randomly choose between the tied ideas and derive the correct index in reference
                    # to the eff_inv_in_period array that contains all ideas that will ever be created 
                    # instead of only this scientist's available ideas that idx_max_return is in reference to
                    idea_choice = int(np.random.choice(idx_second_max)) + [(model.schedule.time + 1)*(model.ideas_per_time)] - len(final_returns_avail_ideas)
                    return(idea_choice, second_max)
                # If there are no ties for the second highest return just return the only index
                else:
                    idea_choice = idx_second_max[0] + [(model.schedule.time + 1)*(model.ideas_per_time)] - len(final_returns_avail_ideas)
                    return(idea_choice, second_max)
        return(idea_choice, max_return)
    
    
class Scientist(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        # Scalar: amount of effort a scientist starts with in each time period
        # he or she is alive (not accounting for start effort decay)
        self.start_effort = poisson(lam=10)

        # Scalar: rate of decay for start_effort of old scientists; currently
        # set at 1 but can be adjusted as necessary
        self.start_effort_decay = 1

        # Scalar: amount of effort a scientist has left; goes down within a
        # given time period as a scientist invests in various ideas
        self.avail_effort = self.start_effort
        
        # Array: investment cost for each idea for a given scientist; a scientist
        # must first pay an idea's investment cost before receiving returns from
        # additional investment
        self.k = poisson(lam=2, size=model.total_ideas)
        
        # Arrays: parameters determining perceived returns for ideas, which are
        # distinct from true returns. Ideas are modeled as logistic CDFs ("S" curve)
        self.sds = poisson(4, model.total_ideas)
        self.means = poisson(25, model.total_ideas)
        # Ensures that none of the standard devs are equal to 0
        # NOTE: May be worth asking Jay if there is a better way to handle this
        self.sds += 1

        # Create the ideas/returns matrix
        self.returns_matrix = \
            create_return_matrix(model.total_ideas, max(model.max_investment), self.sds, self.means)
        
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

        
    def step(self):
        # Check a scientist's age in the current time period
        # NOTE: model time begins at 0
        # NOTE: +3 ensures that we correctly calculate the current age; see
        # notes on time periods and ages in documentation
        self.current_age = (self.model.schedule.time - self.birth_order) + 3

        # Young scientist
        if self.current_age == 0:       
            # Array: has length equal to the total number of ideas in the model,
            # with 0s, 1s, 2s, etc. that indicate which time periods ideas are
            # from
            idea_periods = np.arange(self.model.total_ideas)//self.model.ideas_per_time
            
            # Array: Contains 1s for ideas in the current time period and previous
            # time period, indicating that a young scientist can invest in those
            # ideas. The rest of the array has 0s
            self.avail_ideas = np.logical_or(idea_periods == self.model.schedule.time, \
                idea_periods == (self.model.schedule.time - 1))
            
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
                self.perceived_returns[self.idea_choice] += self.max_return

        # Old scientist
        if self.current_age == 1:       
            # Array: has length equal to the total number of ideas in the model,
            # with 0s, 1s, 2s, etc. that indicate which time periods ideas are
            # from
            idea_periods = np.arange(self.model.total_ideas)//self.model.ideas_per_time
            
            # Array: Contains 1s for ideas in the current time period and previous
            # two time periods, indicating that an old scientist can invest in those
            # ideas. The rest of the array has 0s
            self.avail_ideas = np.logical_or(np.logical_or(idea_periods == self.model.schedule.time, \
                idea_periods == (self.model.schedule.time - 1)), idea_periods == (self.model.schedule.time - 2))
            
            # Determine start/available effort based on the rate of decay for old scientists
            # Currently, start_effort_decay is set to 1, meaning there is no decay for
            # old scientists
            self.start_effort = self.start_effort - self.start_effort_decay * self.current_age
            self.avail_effort = self.start_effort
        
            # Reset effort invested in the current time period
            self.eff_inv_in_period[:] = 0
            
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
                self.perceived_returns[self.idea_choice] += self.max_return
        
        else:
            
            # Scientists are alive and able to do things for only two time periods
            # (i.e. can't do anything when they are not 0 or 1)
            pass
        


class ScientistModel(Model):
    def __init__(self, N, ideas_per_time, time_periods):
        
        # Scalar: indicates the total number of scientists in the model
        # N is the number of scientists per time period
        self.num_scientists = N * time_periods
        
        # Scalar: number of ideas unique to each time period
        self.ideas_per_time = ideas_per_time
        
        # Scalar: total number of ideas in the model. +2 is used to account
        # for first two, non-steady state time periods
        self.total_ideas = ideas_per_time * (time_periods + 2)
        
        # Array: stores the max investment allowed for each idea
        self.max_investment = poisson(lam=10, size=self.total_ideas)
        
        # Array: store parameters for true idea return distribution
        self.true_sds = poisson(4, size=self.total_ideas)
        self.true_means = poisson(25, size=self.total_ideas)
        
        # Ensures that none of the standard devs are equal to 0
        self.true_sds += 1
        
        # Array: keeps track of total effort allocated to each idea across all
        # scientists
        self.total_effort = np.zeros(self.total_ideas)
        
        # Make scientists choose ideas and allocate effort in a random order
        # for each step of the model (i.e. within a time period, the order
        # in which young and old scientists get to invest in ideas is random)
        self.schedule = RandomActivation(self)
        for i in range(4, self.num_scientists + 4):
            a = Scientist(i, self)
            self.schedule.add(a)
        
        # Create data collector method for keeping track of variables over time
        self.datacollector = DataCollector(
            model_reporters = {"Total effort": "total_effort"},
            agent_reporters = {"Effort invested": "effort_invested", "Perceived returns": "perceived_returns"})
        
    def step(self):
        # Call data collector to keep track of variables at each model step
        self.datacollector.collect(self)
        self.schedule.step()
