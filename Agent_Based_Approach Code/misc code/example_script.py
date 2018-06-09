from classes import *
import matplotlib.pyplot as plt

# SAMPLE RUN
cycles = 25
ideas_per_cycle = 4
scientists_per_cycle = 2
granularity = 1
total_ideas = cycles * ideas_per_cycle

# Define distribution of standard deviations of return curves
true_means_mean = 50 * np.ones(total_ideas)
true_means_std_dev = 5 * np.ones(total_ideas)

# Define distribution of standard deviations of return curves
true_std_devs_mean = 1.2 * np.ones(total_ideas)
true_std_devs_std_dev = 0.2 * np.ones(total_ideas)

# Define distribution of starting effort (randomness built into class)
starting_effort_mean = 10
starting_effort_std_dev = 1

# Define distribution of k's for each scientists for each idea
k_mean = 5 * np.ones(total_ideas)
k_std_dev = 0.3 * np.ones(total_ideas)

###### Define "truth" variables #######

# Create true means for return curves
true_means = np.random.normal(true_means_mean, true_means_std_dev)

# Create true std devs for return curves
true_std_devs = np.random.normal(true_std_devs_mean, true_std_devs_std_dev)


max_idea_effort = 10000000000 * np.ones(total_ideas)
model = ScientistModel(scientists_per_cycle, ideas_per_cycle, cycles, granularity, max_idea_effort, \
                    	true_means_mean, true_means_std_dev, true_std_devs_mean, true_std_devs_std_dev, \
                    	starting_effort_mean, starting_effort_std_dev, k_mean, k_std_dev)
for i in range(cycles):
    print(' ')
    print("CYCLE ", i)
    print(' ')
    model.step()


##### PLOTTING #####
# idea_effort_invested = random.choice(model.schedule.agents).current_idea_effort
# objects = range(len(idea_effort_invested))
# y_pos = np.arange(len(objects))
# plt.bar(y_pos, idea_effort_invested, align='center', alpha = 0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Total Effort')
# plt.title('Total Effort invested in each Idea across all Scientists')
# plt.xlabel('Idea Number')

# true_returns = norm.cdf(idea_effort_invested, true_means, true_std_devs)
# objects = range(len(idea_effort_invested))
# y_pos = np.arange(len(objects))
# plt.bar(y_pos, idea_effort_invested, align='center', alpha = 0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Total Returns')
# plt.title('Total Returns from each Idea across all Scientists')
# plt.xlabel('Idea Number')
# plt.show()


