NECESSARY INSTALLATIONS:

Anaconda: 
	
	Windows: https://docs.continuum.io/anaconda/install-windows.html
	OS X: https://docs.continuum.io/anaconda/install-macos

Github Desktop:
	
	https://desktop.github.com/

FIRST STEPS:

- Go to the Command Prompt/Terminal and install the mesa package using "pip install mesa"
- Clone the Scientist_Simulation code from Github using this link
	https://github.com/kmusen/Scientist_Simulation-.git
- This can be done most easily through the graphical interface Github Desktop

OPENING THE CODE:

- Open the application "Jupyter Notebook" that was installed when Anaconda was installed. 

- A browser/server will open that will list all the files in your home directory

- Navigate to the file that the github repository was cloned to 

- The code for the agent-based approach will be located in Scientist_Simulation-/Agent_Based_Approach Code

- Open up the "Scientist Simulation.ipynb" file by clicking on the name

EXPLANATION OF CODE FORMAT:

- The first cell is the main code that defines the Scientist class and the Agent-Based model class

	- The scientist is defined as a normal class 
	- The "step" function defines what the scientist does at every time point

	- The "ScientistModel" class defines everything that the simulation needs to know to run
	- The "step" function for this class defines what happens at every time point

	
- A sample simulation run is the second cell in the Jupyter Notebook

- The third cell is the current state of the model

RUNNING THE SIMULATION:

- You can run the sample simulation by clicking "Cells" on the menu bar at the top of the page, and clicking "Run all cells"

	- The output will be beneath the code that was just run

	