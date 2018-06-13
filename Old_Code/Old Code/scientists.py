"""
Toy model with scientists
"""

import simpy 
import random

env = simpy.Environment()


class Scientist(object):

	def __init__(self, birthyear, env):
		self.env = env
		self.birthyear = int(birthyear)
		self.age = "young"
		self.process = env.process(self.study())
		self.ideas_can_study = []
		self.effort_available = 1.0
		self.k_used = 0.0

	def __str__(self):
		s = "scientist" + str(self.birthyear)
		return s

	def get_old(self):
		self.age = "old"
	
	def dies(self):
		self.age = "dead"

	def is_dead(self):
		if self.age == "dead":
			return True
		else:
			return False

	def get_age(self):
		return self.age

	def get_birthyear(self):
		return self.birthyear

	def add_effort_available(self):
		self.effort_available =+ 1
	
	def use_k(self, idea_num):

		self.ideas_can_study.append(ideas_list[int(idea_num)])
		self.effort_available -= k
		self.k_used += 1.0

	def get_k(self):

		return self.k_used

	def get_study_list(self):

		return self.ideas_can_study
	
	def study(self):
		while True:

			options = [0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0] #creates options for effort for first project
			
			options_minus_k = []
			
			for i in range(len(options)):
				if options[i] <= (1.0-k):
					options_minus_k.append(options[i])

			if self.age == "dead": #so that we know which scientists are no longer doing research

				print "Scientist " +str(self.birthyear) + " is dead" 

			elif self.birthyear ==0 and self.age == "young": #Scientist 0 in year 0 can only work on idea 0
				
				self.use_k(0)

				current = self.effort_available
				
				print "\nIn time 0:\n"
				print "Scientist 0 exerts " + str(current) + " units of effort on Idea 0"

				self.effort_available = 0.0

				ideas_list[0].add_effort(current)
 
			elif self.birthyear == 0 and self.age == "old": #Scientist 0 in year 1 only has 2 options instead of 3 like other old scientists
				
				self.add_effort_available()

				past = random.choice(options)
				if 1.0 - past <= k:
					past = 1

				self.effort_available -= past

				
				if past != 1.0:
					self.use_k(1)
					current = 1.0 - past - k
				else:
					current = 0.0

				self.effort_available -= current 
				
				print "Scientist 0 exerts " + str(past) + " units of effort on idea 0 and " + str(current) + " units of effort on idea 1"

				ideas_list[self.birthyear].add_effort(past)
				ideas_list[self.birthyear+1].add_effort(current)

			elif self.age == "young": #randomly splits 1 unit of effort between two options given to young scientists
				
				current = random.choice(options_minus_k) 
				if current != 0.0:
					self.use_k(self.birthyear)
					self.effort_available -= current

					if int((self.effort_available*10)) <= int((10*k)):
						current = 1.0 - k
						self.effort_available = 0 
						#print  "e1" + str(self.effort_available)

				if current != (1.0-k):
					
					self.use_k((self.birthyear-1))
					#print "e3" + str(self.effort_available)

					
					grandfather = self.effort_available 
				
				else:

					grandfather = 0.0

				self.effort_available -=grandfather

				print "\nIn time " +str(self.birthyear) + ":\n"
				print "Scienitst " +str(self.birthyear) + " exerts " + str(grandfather) + " units of effort on idea " + str(self.birthyear-1) + " and " + str(current) + " units of effort on idea " + str(self.birthyear)

				ideas_list[self.birthyear-1].add_effort(grandfather)
				ideas_list[self.birthyear].add_effort(current)

			else: #randomly splits effort between three options given to old scientists 

				self.add_effort_available()
				
				current =random.choice(options_minus_k)

				if current > 0.0:
					self.use_k((self.birthyear+1))
					self.effort_available -= current
					if self.effort_available <= k:
						current = 1.0-k
						self.effort_available = 0.0


				if current == (1.0-k):
					grandfather =0.0
					past = 0.0
				

				else:


					remaining = self.effort_available

					iterations = int(remaining / .1)

					#print "wolf"
					
					if self.effort_available==.1:
						options2 = [0.0, .1]
					
						#print "penguin"
					
					else:
						options2 = [0.0]
						for i in range(1, (iterations+1)):
							options2.append((.1*i))
					
						#print "elephant"
					
					if ideas_list[(self.birthyear-1)] in self.ideas_can_study:


						grandfather = random.choice(options2)

						self.effort_available -= grandfather

						#print "monkey"

						#print current
						
						if (current > 0.0 and current+grandfather == 1-k):
							
								past =0.0

								#print "cat"
						
						elif int(grandfather) == 1:

							past = 0.0 

						
							#print "fish"
						
						else:

							#print "strawberry"
							if ideas_list[(self.birthyear)] in self.ideas_can_study:
								
								past = self.effort_available
								self.effort_available = 0
							
								#print "apple"
							else:

								#print "bannana"
								
								if int(10*self.effort_available) <= int(k*10):
									grandfather += self.effort_available
									past = 0
									self.effort_available = 0

									#print "kiwi"
								
								else:	
									past = self.effort_available - k
									self.effort_available = 0
									#print "carrot"
					else:

						#print "salad"
						if int(10*self.effort_available) <= int(k*10):
							
							#print "soup"
							if ideas_list[(self.birthyear)] in self.ideas_can_study:
								grandfather = 0
								past = self.effort_available
								self.effort_available = 0
						
								#print "sushi"
						else:

							options3 = []
							iterations3 = int(self.effort_available*10)

							for i in range(1, (iterations3+1)):
								options3.append((.1*i))
							

							val = random.choice(options3)
							
							if val > 0:

								if ideas_list[(self.birthyear-1)] in self.ideas_can_study:
									grandfather = val
									self.effort_available -= grandfather

								else:

									self.use_k(self.birthyear-1)
									grandfather = val - k
									self.effort_available -= grandfather

								if int(10*self.effort_available) <= int(k*10):
									grandfather += self.effort_available
									past =0.0
									self.effort_available = 0.0
								else:
									self.use_k(self.birthyear)
									past = self.effort_available
									self.effort_available = 0

							else:

								grandfather = 0.0
								
								if ideas_list[(self.birthyear)] in self.ideas_can_study:
									past = self.effort_available
									self.effort_available = 0.0
								
								else:
									self.use_k(self.birthyear)
									past = self.effort_available - k
									self.effort = 0.0

							

					
				#print "hallo"
				ideas_list[self.birthyear-1].add_effort(grandfather)
				ideas_list[self.birthyear].add_effort(past)
				ideas_list[self.birthyear+1].add_effort(current)
				
				print "Scientist " + str(self.birthyear) + " uses " + str(current) + " units of effort on idea " +str(self.birthyear+1) + ", " + str(past) + " units of effort on idea " + str(self.birthyear) + ", and " +str(grandfather) + " units of effort on idea " +str(self.birthyear-1)
			
			t = 1
			yield env.timeout(t)
				
class Idea(object):
	
	def __init__(self, year_created):
		self.year_created = int(year_created)
		
		self.contributors = []

		self.total_effort = 0.0

	def __str__(self):
		s = "idea" +str(self.year_created)
		return s
	
	def add_effort(self, amount): #method to add effort into an idea
		
		self.total_effort += float(amount)

	def get_effort(self): #method to find out how much effort has been put into an idea

		return self.total_effort
		

def lifecycle(env): #creates, ages, and kills scientists 
	
	global researchers

	researchers = []
	counter = 0
	researchers.append((Scientist(0, env)))
	time = 1
	yield env.timeout(time)

	#num_researchers = 1 #we start with one scientists in year 0

	while True:
		counter += 1
		researchers.append(Scientist((counter), env)) #adds scientists
		researchers[int(counter-1)].get_old() #ages scientists 
		if len(researchers) > 2:
			researchers[counter-2].dies() #kills scientists 
		duration = 1
		#for researcher in researchers:
			#print researcher

		yield env.timeout(duration)



def main():

	cycles = int(raw_input("How many generations of scientists do you want?"))

	global k 

	k = float(raw_input("What is the k?"))

	global ideas_list #allows access to this list within the classes 

	ideas_list = []

	for i in range(cycles): #creates the number of ideas specified by the number of cycles the user wants 
		ideas_list.append(Idea(i))

	env.process(lifecycle(env))

	env.run(until = cycles) #runs simulation

	total_effort_all_ideas =0

	for i in range(cycles):

		print str(ideas_list[i].get_effort()) + " units total effort on idea " + str(i)

		total_effort_all_ideas += ideas_list[i].get_effort() #adds up total effort into all ideas, serves as a check for bugs

	print str(total_effort_all_ideas) + " units effort on all ideas combined"

	for i in range(len(researchers)):
		print str(researchers[i].get_k()) + str(researchers[i].get_study_list())


main()

		
	
	

	

