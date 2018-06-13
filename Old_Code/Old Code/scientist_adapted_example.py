"""

Example adpated from car example code on documentation site

"""



def scientist(env):
	
	print "Scientist 1 is born at time 1"
	young = 1
	yield env.timeout(young)
	
	counter = -1
	
	scientists = 1
	while True:
		counter += 2
		print "Scientist " + str(counter) + " gets old at time " + str(env.now +1)
		print"Scientist " + str(counter+1) + " is born at time " + str(env.now + 1)
		scientists += 1
		duration = 1
		if env.now == 14:
			 print "There are " + str(scientists) + " scientists"
		yield env.timeout(duration)

		print "Scientist " + str(counter) + " dies at time " +str(env.now +1)
		print "Scientist " + str(counter+1) +" gets old at time " + str(env.now +1)
		print "Scientist " +str(counter+2) + " is born at time " + str(env.now +1)
		scientists +=1
		duration2 = 1
		if env.now ==14:
			print "There are "+  str(scientists) + " scientists"
		yield env.timeout(duration2)

import simpy 
env = simpy.Environment()
env.process(scientist(env))

env.run(until=15)

 				
