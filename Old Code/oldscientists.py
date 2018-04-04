"""
Toy model with scientists
"""

class Scientist(object):

	def __init__(self, birthyear):
		self.birthyear = int(birthyear)
		self.age = "young"

	def __str__(self):
		s = "scientist" + str(self.birthyear)
		return s

	def get_old(self):
		self.age = "old"
	
	def die(self):
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

class Idea(object):
	
	def __init__(self, year_created):
		self.year_created = int(year_created)
		
		self.contributors = []

		self.total_effort = 0.0

	def __str__(self):
		s = "idea" +str(self.year_created)
		return s
	
	def add_contributor(scientist, amount):

		self.contributors.append(scientist)
		
		sef.total_effort += float(amount)
		
def main():
	
	scientists = []
	
	generations = 0
	
	print "\nLet's create scientists! \n"
	for i in range(5):

		scientists.append(Scientist((i+1)))
		
		print "created " + str(scientists[i])
		
		generations += 1
	
		if generations > 1:
			scientists[(i-1)].get_old()

		if generations > 2:
			scientists[(i-2)].die()			
	
	print "\nAt year " + str(generations) + "...\n"

	for k in range(len(scientists)):
		print str(scientists[k]) + " is " + scientists[k].get_age()
	

		
	ideas = []
	
	print "\nLet's create ideas! \n"

	for i in range(5):

		ideas.append(Idea(i+1))

		print ideas[i]

	print ""


	
	

	
main()	
