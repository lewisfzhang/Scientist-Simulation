Scientist Simulation README:

Pre-requisites:
-Python 2.7 installed
-Python packages required:
	1. itertools
	2. enum
	3. matplotlib.pyplot
	4. numpy
	5. scipy.stats
	6. math

** Note: All relevant code is located in the "Current Code" directory

DESCRIPTION OF FILES:

1. optimization_equations.py

Contains functions that will graph the return function, and also return a value from a curve created from inputs (standard deviation, mean). 

2. possible_effort_splits.py

Contains functions that deal with creating all possible effort splits for young and old scientists. 

3. maximization.py

Main file that runs the simulation. Uses optimization_equations.py and possible_effort_splits.py functions in order to run the simulation. 

Contains functions that will calculate returns from old/young scientist splits and the return function defined. 


HOW TO RUN: 

-The python file "maximization.py" takes in exactly 12 arguments
-Arguments:
	1. size of the effort unit/granularity

	2. k for the young scientist

	3. k for the old scientist

	4. total effort allowed to each scientist

	5. how many decimal places the granularity has

	6-7. the initial young effort split

	8-10. the initial old effort split

	11. how many cycles to run

	12. standard deviation to vary S-shape of the return function

- ** NOTES: 
	1. The s-shape return function originally discussed requires a standard deviation of 1.2

	2. The young split (0.3, 0.3) == (t, t+1) and old split (0.1, 0.1, 0.6) = (t-1, t, t+1) are the best splits found for running the simulation. This many be updated in the future with experimentation

-Examples:
Command: python 0.01 0.1 0.1 1.0 2 0.3 0.3 0.1 0.1 0.6 2 1.2

Explanation of Arguments in Order: 
1. the granularity of effort units is 0.01

2. The k for the young and old scientists are both 0.1

3. each scientist is allowed 1.0 total effort

4. the granularity goes up to 2 decimal places (i.e. if the first argument was 0.001, this argument would be 3)

5. (0.3, 0.3) == (effort for current idea, effort for new idea) for young scientist

6. (0.1, 0.1, 0.6) == (old idea effort, current idea effort, new idea effort) for the old scientist

7. the simulation will be run for 2 time periods

8. the standard deviation of the S-shape return function is 1.2


