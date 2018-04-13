from mesa import Agent, Model
from mesa.time import BaseScheduler
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
# 2) effort_avail_ideas - array that contains cumulative effort already expended by all scientists on avail ideas
# 3) marginal_effort - array of "equivalent" marginal efforts, which equals
#    the max, current investment cost plus one minus individual investment costs for available ideas
#
# Output: array of dim = avail_ideas that has associated marginal returns
def calc_cum_returns(scientist):
    final_returns_avail_ideas = []
    for idea in scientist.avail_ideas:
        start_index = scientist.total_effort[idea]
        stop_index = start_index + scientist.marginal_effort[idea]
        returns = scientist.returns_matrix[idea, np.arange(start_index, stop_index)]
        total_return = sum(returns)
        final_returns_avail_ideas.append(total_return)
    return(np.asarray(final_returns_avail_ideas))

class Scientist(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        # Scalar: amount of effort a scientist is born with (M)
        self.start_effort = poisson(lam=10)

        # Scalar: rate of decay for starting_effort (lambda_e)
        self.start_effort_decay = 1

        # Scalar: amount of effort a scientist has left
        self.avail_effort = self.start_effort.copy() 
        
        # Investment cost for each idea for the scientist
        self.k = poisson(lam=2, size=model.total_ideas)
        
        # Parameters determining perceived returns for ideas
        self.sds = poisson(4, model.total_ideas)
        self.means = poisson(50, model.total_ideas)
        
        # Create the ideas/returns matrix
        self.returns_matrix = \
            create_return_matrix(model.total_ideas, max(model.max_investment), sds, means)
        
        # Records when the scientist was born
        self.birth_time = model.schedule.time
        
        # Array keeping track of how much effort this scientist has invested in each idea
        self.effort_invested = np.zeros(model.total_ideas)
        
        # Array to keep track of which ideas from which time periods can be worked on
        self.avail_ideas = np.zeros(model.total_ideas)
        
        # Create array to keep track of total effort allocated to each idea
        self.total_effort = model.total_effort
        
    def step(self):
        print(self.returns_matrix)

        # Check scientist's age in the current time period
        self.current_age = model.schedule.time - self.birth_time

        if self.current_age == 0:       # Young scientist
            idea_periods = np.arange(model.total_ideas)//model.ideas_per_time
            
            # Can work on ideas in the current or previous time period
            self.avail_ideas = np.logical_or(idea_periods == model.schedule.time, \
                idea_periods == (model.schedule.time - 1))
            
            ##### HEURISTIC FOR CHOOSING WHICH IDEA TO INVEST IN #####
            
            # Determine if scientists have yet to pay investment costs
            # Whether a scientist has invested no effort into an idea
            no_effort_inv = (self.effort_invested == 0)
            # Calculate current investment costs based on prior investments
            # Matrix has invest costs or 0 if scientist has already invested effort
            self.curr_k = no_effort_inv * self.k
            
            # Want to pull returns for expending self.increment units of effort,
            # where increment equals the max invest cost across all ideas that
            # haven't been invested in yet plus 1
            self.increment = max(self.curr_k) + 1
            
            # Array with equivalent "marginal" efforts across ideas
            # For idea with the max invest cost, marginal_effort will equal 1
            # All others - marginal_effort is equal to increment minus invest cost, if any
            self.marginal_effort = self.increment - self.curr_k
            
            # Array (dim = num(avail_ideas)x1) that has the cumulative returns for increment efforts in ideas
            self.cum_returns = calc_cum_returns(self)
                        
            # Select idea with the maximum, true return

            
            ### NOTE: Need to program edge case when multiple ideas have the same return
            ### NOTE: Need to factor in when ideas hit maximum effort invested
            
        
        if self.current_age == 1:       # Old scientist
            idea_periods = np.arange(model.total_ideas)//model.ideas_per_time
            
            # Can work on ideas in the current or previous two periods
            self.avail_ideas = np.logical_or(np.logical_or(idea_periods == model.schedule.time, \
                idea_periods == (model.schedule.time - 1)), idea_periods == (model.schedule.time - 2))
            
            # Determine how much effort a scientist has in the current time period
            self.curr_period_effort = self.start_effort - self.start_effort_decay * self.current_age

        else:
            # Scientists are alive and able to do things for only two time periods
            pass


#        # Determine how much effort the scientist has in current time period
#        self.curr_start_effort = self.start_effort - self.start_effort_decay*model.schedule.time
#        print(self.unique_id)
#        print(self.curr_start_effort)
#        if self.wealth == 0:
#            return
#        other_agent = random.choice(self.model.schedule.agents)
#        other_agent.wealth += 1
#        self.wealth -= 1

class ScientistModel(Model):
    def __init__(self, N, ideas_per_time, time_periods):
        self.num_scientists = N
        
        self.total_ideas = ideas_per_time * time_periods
        
        # Store the max investment allowed in any idea
        self.max_investment = poisson(lam=50, size=self.total_ideas)
        
        # Store parameters for true idea return distribution
        self.true_sds = poisson(4, size=self.total_ideas)
        self.true_means = poisson(50, size=self.total_ideas)

        # Create array to keep track of total effort allocated to each idea
        self.total_effort = np.zeros(self.total_ideas)
        
        self.schedule = BaseScheduler(self)
        for i in range(self.num_scientists):
            a = Scientist(i, self)
            self.schedule.add(a)
            
    def step(self):
        self.schedule.step()