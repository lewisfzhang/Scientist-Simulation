class Idea(object):
	
	def __init__(self, year_created):
		self.year_created = int(year_created)
		self.total_effort = 0.0

	def __str__(self):
		s = "idea" +str(self.year_created)
		return s
	def get_year_created(self):
		return self.year_created

	def add_effort(self, amount): #method to add effort into an idea
		self.total_effort += float(amount)

	def get_effort(self): #method to find out how much effort has been put into an idea
		return self.total_effort

class Scientist(object):

	def __init__(self, prng, year_created, total_effort):
		self.age = age

	def __str__(self):
		s = str(self.age) + " scientist"
		return s

	def get_age(self):
		return self.age

