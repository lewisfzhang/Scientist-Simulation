# To Do List:
# 1) Update old scientist code - DONE
# 2) Figure out how to code base case (time period 0)
# 3) Code edge case in which 80% of effort has been invested in only 1 idea - DONE
# 4) Double check with Michelle that model needs to be passed to agent step
# 5) Ask Michelle about referencing model class
# 6) TEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#


from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import numpy as np
from numpy.random import poisson, logistic

# Sigmoid function
def logistic_cdf(x, loc, scale):
    return 1/(1+np.exp((loc-x)/scale))

# Sigmoid function starting at x = 0, scale, shape > 0, x >= 0
def gompertz_cdf(x, shape, scale):
    return 1-np.exp(-shape*(np.exp(scale*x)-1))

def create_return_matrix(num_ideas, max_of_max_inv, sds, means):
    x = np.arange(max_of_max_inv + 1)
    returns_list = []
    for i in range(num_ideas):
        returns = logistic_cdf(x, loc = means[i], scale = sds[i])
        # Stacks arrays horizontally
        to_subt_temp = np.hstack((0,returns[:-1]))
        # Calculates return per unit of effort
        returns = returns - to_subt_temp
        returns_list.append(returns)
    return(np.array(returns_list))

# Function to calculate "marginal" returns for available ideas, taking into account invest costs
# Input:
# 1) avail_ideas - array that indexes which ideas are available to a given scientist
# 2) total_effort - array that contains cumulative effort already expended by all scientists
# 3) max_investment - array that contains the max possible investment for each idea
# 4) marginal_effort - array of "equivalent" marginal efforts, which equals
#    the max, current investment cost plus one minus individual investment costs for available ideas
#
# Output:
# 1) idx_max_return - scalar that is the index of the idea the scientist chose to invest in
# 2) max_return - scalar that is the perceived return of the associated, chosen idea
    
def calc_cum_returns(scientist, model):
    final_returns_avail_ideas = []
    # Limit on the amount of effort that a scientist can invest in a single idea
    # in one time period
    invest_cutoff = round(scientist.start_effort * 0.8)
    
    for idea in np.where(scientist.avail_ideas == True)[0]:
        # OR Conditions
        # 1st) Edge case in which scientist doesn't have enough effort to invest in
        # an idea given the investment cost
        # 2nd) Ensures that effort invested in ideas doesn't go over max invest
        # 3rd) Prevents scientists from investing further in an idea if the
        # scientist has already invested at or above the investment cutoff
        effort_left = scientist.max_investment[idea] - model.total_effort[idea]
        if scientist.marginal_effort[idea] <= 0 or scientist.marginal_effort[idea] > effort_left or \
            scientist.eff_inv_in_period[idea] >= invest_cutoff:
            final_returns_avail_ideas.append(0)
        else:
            start_index = model.total_effort[idea]
            stop_index = start_index + scientist.marginal_effort[idea]
            returns = scientist.returns_matrix[idea, np.arange(start_index, stop_index)]
            total_return = sum(returns)
            final_returns_avail_ideas.append(total_return)
    
    max_return = max(final_returns_avail_ideas)
    idx_max_return = np.where(final_returns_avail_ideas == max_return)[0]
    # Resolves edge case in which there are multiple max returns
    if len(idx_max_return > 1):
        return(np.random.choice(idx_max_return)[0], max_return)
    else:
        return(idx_max_return[0], max_return)

class Scientist(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        # Scalar: amount of effort a scientist is born with (M)
        self.start_effort = poisson(lam=10)

        # Scalar: rate of decay for starting_effort (lambda_e)
        self.start_effort_decay = 1

        # Scalar: amount of effort a scientist has left
        self.avail_effort = self.start_effort
        
        # Investment cost for each idea for the scientist
        self.k = poisson(lam=2, size=model.total_ideas)
        
        # Parameters determining perceived returns for ideas
        self.sds = poisson(4, model.total_ideas)
        self.means = poisson(25, model.total_ideas)
        
        # Create the ideas/returns matrix
        self.returns_matrix = \
            create_return_matrix(model.total_ideas, max(model.max_investment), sds, means)
        
        # Records when the scientist was born
        self.birth_time = model.schedule.time
        
        # Array keeping track of how much effort this scientist has invested in each idea
        self.effort_invested = np.zeros(model.total_ideas)
        
        # Array to keep track of which ideas from which time periods can be worked on
        self.avail_ideas = np.zeros(model.total_ideas)
        
        # NOTE: Commented out because total effort should be a model-level variable,
        # as it needs to be updated globally for each scientist
        # Create array to keep track of total effort allocated to each idea
        # self.total_effort = model.total_effort
        
        # Create array to keep track of total, perceived returns for each idea
        self.perceived_returns = np.zeros(model.total_ideas)
        
        # Create a copy of the max investment array from model class
        self.max_investment = model.max_investment.copy()
        
        # Array to keep track of effort invested ONLY during the current time period
        self.eff_inv_in_period = np.zeros(model.total_ideas)
        
    def step(self, model):
        print(self.returns_matrix)

        # Check scientist's age in the current time period
        self.current_age = model.schedule.time - self.birth_time

        if self.current_age == 0:       # Young scientist
            idea_periods = np.arange(model.total_ideas)//model.ideas_per_time
            
            # Can work on ideas in the current or previous time period
            self.avail_ideas = np.logical_or(idea_periods == model.schedule.time, \
                idea_periods == (model.schedule.time - 1))
            
            while self.avail_effort > 0:
                ##### HEURISTIC FOR CHOOSING WHICH IDEA TO INVEST IN #####
                
                # Determine whether scientists have invested zero effort in an idea
                no_effort_inv = (self.effort_invested == 0)
                # Calculate current investment costs based on prior investments
                # Matrix has invest costs or 0 if scientist has already invested effort
                self.curr_k = no_effort_inv * self.k
                
                # Want to pull returns for expending self.increment units of effort,
                # where increment equals the max invest cost across all ideas that
                # haven't been invested in yet plus 1
                self.increment = max(self.curr_k) + 1
                # Edge case in which scientist doesn't have enough effort to invest in idea with greatest invest cost
                if self.avail_effort < self.increment:
                    self.increment = self.avail_effort
                
                # Array with equivalent "marginal" efforts across ideas
                # For idea with the max invest cost, marginal_effort will equal 1
                # All others - marginal_effort is equal to increment minus invest cost, if any
                self.marginal_effort = self.increment - self.curr_k
                
                # Selects idea that gives the max return given equivalent "marginal" efforts
                self.idea_choice, self.max_return = calc_cum_returns(self, model)
                
                # Update parameters after idea selection and effort expenditure
                model.total_effort[self.idea_choice] += self.increment
                self.effort_invested[self.idea_choice] += self.increment
                self.eff_inv_in_period[self.idea_choice] += self.increment
                self.avail_effort -= self.increment
                self.perceived_returns[self.idea_choice] += self.max_return
        
        if self.current_age == 1:       # Old scientist
            idea_periods = np.arange(model.total_ideas)//model.ideas_per_time
            
            # Can work on ideas in the current or previous two periods
            self.avail_ideas = np.logical_or(np.logical_or(idea_periods == model.schedule.time, \
                idea_periods == (model.schedule.time - 1)), idea_periods == (model.schedule.time - 2))
            
            # Determine available effort based on rate of decay for old scientists
            # Added the top line so that calc_cum_returns would take into account
            # the reduced start effort for old scientists
            self.start_effort = self.start_effort - self.start_effort_decay * self.current_age
            self.avail_effort = self.start_effort
        
            while self.avail_effort > 0:
                ##### HEURISTIC FOR CHOOSING WHICH IDEA TO INVEST IN #####
                
                # Determine whether scientists have invested zero effort in an idea
                no_effort_inv = (self.effort_invested == 0)
                # Calculate current investment costs based on prior investments
                # Matrix has invest costs or 0 if scientist has already invested effort
                self.curr_k = no_effort_inv * self.k
                
                # Want to pull returns for expending self.increment units of effort,
                # where increment equals the max invest cost across all ideas that
                # haven't been invested in yet plus 1
                self.increment = max(self.curr_k) + 1
                # Edge case in which scientist doesn't have enough effort to invest in idea with greatest invest cost
                if self.avail_effort < self.increment:
                    self.increment = self.avail_effort
                
                # Array with equivalent "marginal" efforts across ideas
                # For idea with the max invest cost, marginal_effort will equal 1
                # All others - marginal_effort is equal to increment minus invest cost, if any
                self.marginal_effort = self.increment - self.curr_k
                
                # Selects idea that gives the max return given equivalent "marginal" efforts
                self.idea_choice, self.max_return = calc_cum_returns(self)
                
                # Update parameters after idea selection and effort expenditure
                model.total_effort[self.idea_choice] += self.increment
                self.effort_invested[self.idea_choice] += self.increment
                self.eff_inv_in_period[self.idea_choice] += self.increment
                self.avail_effort -= self.increment
                self.perceived_returns[self.idea_choice] += self.max_return 
        
        else:
            # Scientists are alive and able to do things for only two time periods
            pass

        # Reset available effort at the end of each time period
        # self.avail_effort = self.start_effort
        
        # Reset effort invested in the current time period
        self.eff_inv_in_period[:] = 0

class ScientistModel(Model):
    def __init__(self, N, ideas_per_time, time_periods):
        self.num_scientists = N
        
        self.total_ideas = ideas_per_time * time_periods
        
        # Store the max investment allowed in any idea
        self.max_investment = poisson(lam=50, size=self.total_ideas)
        
        # Store parameters for true idea return distribution
        self.true_sds = poisson(4, size=self.total_ideas)
        self.true_means = poisson(25, size=self.total_ideas)

        # Create array to keep track of total effort allocated to each idea
        self.total_effort = np.zeros(self.total_ideas)
        
        # Make scientists choose ideas and allocate effort in a random order
        # for each step of the model
        self.schedule = RandomActivation(self)
        for i in range(self.num_scientists):
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