import matplotlib.pyplot as plt

## objects = ('1', '2', '3', '4', '5', '6', '7' ,'8', '9', '10')
#idea_returns = random.choice(model.schedule.agents).current_idea_effort
#objects = range(len(idea_returns))
#y_pos = np.arange(len(objects))
#plt.bar(y_pos, idea_returns, align='center', alpha = 0.5)
#plt.xticks(y_pos, objects)
#plt.ylabel('Total Effort')
#plt.title('Total Effort invested in each Idea across all Scientists')
#plt.xlabel('Idea Number')
#plt.show()

scientist_wealth = [a.wealth for a in model.schedule.agents]
plt.hist(scientist_wealth)
plt.show()