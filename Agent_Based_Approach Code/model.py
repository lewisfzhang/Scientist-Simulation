# To Do List:
# 1) Update old scientist code - DONE
# 2) Figure out how to code base case (time period 0) - DONE
# 3) Code edge case in which 80% of effort has been invested in only 1 idea - DONE
# 4) Double check with Michelle that model needs to be passed to agent step - DONE
# 5) Ask Michelle about referencing model class - DONE
# 6) TEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 7) Need to update code to deal with invest cutoff issue
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
    print("Unique ID: ", scientist.unique_id)
    print("Age of scientist: ", scientist.current_age)
    print("Time period: ", model.schedule.time)
    print("Current investment costs for all ideas: ", scientist.curr_k)
    final_returns_avail_ideas = np.array([])
    # Limit on the amount of effort that a scientist can invest in a single idea
    # in one time period
    # 5/11: Set at 1 because code to deal with invest cutoff edge case not in place
    invest_cutoff = round(scientist.start_effort * 1)

    
    for idea in np.where(scientist.avail_ideas == True)[0]:
        # OR Conditions
        # 1st) Edge case in which scientist doesn't have enough effort to invest in
        # an idea given the investment cost
        # 2nd) Ensures that effort invested in ideas doesn't go over max invest
        # 3rd) Prevents scientists from investing further in an idea if the
        # scientist has already invested at or above the investment cutoff
        print("idea #: ", idea)
        print("effort_left_in_idea:", scientist.effort_left_in_idea[idea])
        if scientist.marginal_effort[idea] <= 0 or scientist.effort_left_in_idea[idea] == 0 or \
            scientist.eff_inv_in_period[idea] >= invest_cutoff:
            print("Amount invested in idea in period:", scientist.eff_inv_in_period[idea])
            print("marginal effort:", scientist.marginal_effort[idea])
#            print("invest cutoff:", invest_cutoff)
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
        else:
            print("No restrictions on investing in this idea")
            start_index = int(model.total_effort[idea])
            stop_index = int(start_index + scientist.marginal_effort[idea])
            returns = scientist.returns_matrix[idea, np.arange(start_index, stop_index)]
            total_return = sum(returns)
            final_returns_avail_ideas = np.append(final_returns_avail_ideas, total_return)
    
    print("avail_effort: ", scientist.avail_effort)           
    print("final_returns_avail_ideas: ", final_returns_avail_ideas)
    print("Current investment costs: ", scientist.curr_k[np.where(scientist.avail_ideas == True)[0]])
    print("Increment: ", scientist.increment)
    max_return = max(final_returns_avail_ideas)
    print("max_returns: ", max_return)
    print(np.where(final_returns_avail_ideas == max_return))
    idx_max_return = np.where(final_returns_avail_ideas == max_return)[0]
    print("idx_max_return: ", idx_max_return)
    # Resolves edge case in which there are multiple max returns
    print("")
    if len(idx_max_return > 1):
        return(np.random.choice(idx_max_return) + [(model.schedule.time + 1)*(model.ideas_per_time)] - len(final_returns_avail_ideas), max_return)
    else:
        return(idx_max_return[0] + [(model.schedule.time + 1)*(model.ideas_per_time)] - len(final_returns_avail_ideas), max_return)
    
    
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
        # Ensures that none of the standard devs are equal to 0
        self.sds += 1
        self.means = poisson(25, model.total_ideas)
        
        # Create the ideas/returns matrix
        self.returns_matrix = \
            create_return_matrix(model.total_ideas, max(model.max_investment), self.sds, self.means)
        
        # Records when the scientist was born; unique_id allows us to select
        # a subset of scientists to be "alive" in each time period
        self.birth_order = unique_id
        
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
        
        # Allows scientists to access model variables
        self.model = model

        
    def step(self):
        
        # Check scientist's age in the current time period
        # NOTE: model time begins at 0
        # NOTE: +3 ensures that we correctly calculate the current age, given
        # that scientists aren't born in time periods 0 and 1
        self.current_age = (self.model.schedule.time - self.birth_order) + 3

        if self.current_age == 0:       # Young scientist
            idea_periods = np.arange(self.model.total_ideas)//self.model.ideas_per_time
            
            # Can work on ideas in the current or previous time period
            self.avail_ideas = np.logical_or(idea_periods == self.model.schedule.time, \
                idea_periods == (self.model.schedule.time - 1))
            
            while self.avail_effort > 0:
                ##### HEURISTIC FOR CHOOSING WHICH IDEA TO INVEST IN #####
                
                # Determine whether scientists have invested zero effort in an idea
                no_effort_inv = (self.effort_invested == 0)
                # Calculate current investment costs based on prior investments
                # Matrix has invest costs or 0 if scientist has already invested effort
                self.curr_k = no_effort_inv * self.k
                
                # Array (size = model.total_ideas) of how much more effort can 
                # be invested in a given idea based on the max investment for
                # that idea and how much all scientists have already invested
                self.effort_left_in_idea = self.max_investment - self.model.total_effort

                # Change current investment cost to 0 if a given idea has 0
                # effort left in it. This prevents the idea from affecting
                # the marginal effort
                for idx, value in enumerate(self.effort_left_in_idea):
                    if value == 0:
                        self.curr_k[idx] = 0
                
                # Want to pull returns for expending self.increment units of effort,
                # where increment equals the max invest cost across all ideas that
                # haven't been invested in yet plus 1
                self.increment = max(self.curr_k[self.avail_ideas]) + 1
                # Edge case in which scientist doesn't have enough effort to invest in idea with greatest invest cost
                if self.avail_effort < self.increment:
                    self.increment = self.avail_effort
                
                # Array with equivalent "marginal" efforts across ideas
                # For idea with the max invest cost, marginal_effort will equal 1
                # All others - marginal_effort is equal to increment minus invest cost, if any
                self.marginal_effort = self.increment - self.curr_k
                
                # Selects idea that gives the max return given equivalent "marginal" efforts
                self.idea_choice, self.max_return = calc_cum_returns(self, self.model)
                
                # Accounts for the edge case in which max_return = 0 (implying that a
                # scientist either can't invest in any ideas [due to investment
                # cost barriers or an idea reaching max investment]). Effort
                # is lost and doesn't carry over to the next period
                if self.max_return == 0:
                    self.avail_effort = 0
                    continue
                
                # Accounts for edge case in which scientist chooses to invest in
                # an idea that has less effort remaining than a scientist's 
                # marginal effort
                if self.marginal_effort[self.idea_choice] > self.effort_left_in_idea[self.idea_choice]:
                    self.marginal_effort[self.idea_choice] = self.effort_left_in_idea[self.idea_choice]
                    self.increment = self.curr_k[self.idea_choice] + self.marginal_effort[self.idea_choice]
                
                # Update parameters after idea selection and effort expenditure
                # NOTE: self.avail_effort should be updated by the increment, not
                # by marginal effort. While we don't care about paid investment costs
                # for the other variables, the amount of effort left within a given
                # time period should be affected by paid investment costs.
                self.model.total_effort[self.idea_choice] += self.marginal_effort[self.idea_choice]
                self.effort_invested[self.idea_choice] += self.marginal_effort[self.idea_choice]
                self.eff_inv_in_period[self.idea_choice] += self.marginal_effort[self.idea_choice]
                self.avail_effort -= self.increment
                self.perceived_returns[self.idea_choice] += self.max_return
        
        if self.current_age == 1:       # Old scientist
            idea_periods = np.arange(self.model.total_ideas)//self.model.ideas_per_time
            
            # Can work on ideas in the current or previous two periods
            self.avail_ideas = np.logical_or(np.logical_or(idea_periods == self.model.schedule.time, \
                idea_periods == (self.model.schedule.time - 1)), idea_periods == (self.model.schedule.time - 2))
            
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
                
                # Array (size = model.total_ideas) of how much more effort can 
                # be invested in a given idea based on the max investment for
                # that idea and how much all scientists have already invested
                self.effort_left_in_idea = self.max_investment - self.model.total_effort
                
                # Change current investment cost to 0 if a given idea has 0
                # effort left in it. This prevents the idea from affecting
                # the marginal effort
                for idx, value in enumerate(self.effort_left_in_idea):
                    if value == 0:
                        self.curr_k[idx] = 0
                
                # Want to pull returns for expending self.increment units of effort,
                # where increment equals the max invest cost across all ideas that
                # haven't been invested in yet plus 1
                self.increment = max(self.curr_k[self.avail_ideas]) + 1
                # Edge case in which scientist doesn't have enough effort to invest in idea with greatest invest cost
                if self.avail_effort < self.increment:
                    self.increment = self.avail_effort
                
                # Array with equivalent "marginal" efforts across ideas
                # For idea with the max invest cost, marginal_effort will equal 1
                # All others - marginal_effort is equal to increment minus invest cost, if any
                self.marginal_effort = self.increment - self.curr_k
                
                # Selects idea that gives the max return given equivalent "marginal" efforts
                self.idea_choice, self.max_return = calc_cum_returns(self, self.model)
                
                # Accounts for the edge case in which max_return = 0 (implying that a
                # scientist either can't invest in any ideas [due to investment
                # cost barriers or an idea reaching max investment]). Effort
                # is lost and doesn't carry over to the next period
                if self.max_return == 0:
                    self.avail_effort = 0
                    continue
                
                # Accounts for edge case in which scientist chooses to invest in
                # an idea that has less effort remaining than a scientist's 
                # marginal effort
                if self.marginal_effort[self.idea_choice] > self.effort_left_in_idea[self.idea_choice]:
                    self.marginal_effort[self.idea_choice] = self.effort_left_in_idea[self.idea_choice]
                    self.increment = self.curr_k[self.idea_choice] + self.marginal_effort[self.idea_choice]
                
                # Update parameters after idea selection and effort expenditure
                # NOTE: self.avail_effort should be updated by the increment, not
                # by marginal effort. While we don't care about paid investment costs
                # for the other variables, the amount of effort left within a given
                # time period should be affected by paid investment costs.
                self.model.total_effort[self.idea_choice] += self.marginal_effort[self.idea_choice]
                self.effort_invested[self.idea_choice] += self.marginal_effort[self.idea_choice]
                self.eff_inv_in_period[self.idea_choice] += self.marginal_effort[self.idea_choice]
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
        self.num_scientists = N * time_periods
        
        self.ideas_per_time = ideas_per_time
        
        self.total_ideas = ideas_per_time * (time_periods + 2)
        
        # Store the max investment allowed in any idea
        self.max_investment = poisson(lam=8, size=self.total_ideas)
        
        # Store parameters for true idea return distribution
        self.true_sds = poisson(4, size=self.total_ideas)
        # Ensures that none of the standard devs are equal to 0
        self.true_sds += 1
        
        self.true_means = poisson(25, size=self.total_ideas)

        # Create array to keep track of total effort allocated to each idea
        self.total_effort = np.zeros(self.total_ideas)
        
        # Make scientists choose ideas and allocate effort in a random order
        # for each step of the model
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
        print("Total effort (across scientists) invested in each idea:", self.total_effort)