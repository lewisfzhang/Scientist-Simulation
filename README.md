# Scientist-Simulation
Agent Based Model Simulation

Ways to run simulation:
  1. Via IDE
    - Create virtual environment, install necessary modules
        - pip install (IPython Mesa matplotlib numpy scipy flask requests)
    - Run src/run.py for the whole program
    - If you only want to collect data after running the model, run src/collect.py instead
    - Run src/server.py for interactive session using localhost (browser window will automatically launch upon running)
  2. Via Bash (Preferred)
    - ./src/run.sh
    - Part 1: automatically checks to see if there is a working virtual environment, and install necessary files if not
    - Part 2: saves data as a tar.gz compressed file in data/zipped directory
    - Optional: you can specify 4, 7, or 13 parameters (listed in order with variable name following the description)
        ##### Input Values
        1. Time Periods (time_periods)
        2. Ideas Per Time Period (ideas_per_time)
        3. Number of Scientists Per Time Period (N)
        4. Time Periods Alive (time_periods_alive)
        4/5. True Mean (true_means_lam) [**not included in 7 parameter version!**]

        ##### Proportions
        5/6. Proportion of SDS to Means (prop_sds)
        6/7. Proportion of Start Effort to Means of Idea (prop_means)
        7/8. Proportion of Learning K to Start Effort (prop_start)

        ##### Optimization
        9. What Optimization to Use (switch)
            - 0 = percentiles
            - 1 = z-scores
            - 2 = bayesian stats
            - 3 = greedy heuristic
        10. Void for now, reserved for AI/deep learning optimization later on

        ##### Checks
        11. Whether to Report All Scientists (all_scientists)
        12. Whether to Split Returns Evently Or By Age (use_equal)
        13. Whether to Shift All CDFs / Idea Curves to the Right (use_idea_shift)
        14. Whether We Want Interactive Steps (show_step)
            - **NOTE:** using this makes the program run much longer!
  3. Additional Notes
    - This repository runs on Python 3 and Linux shell
    - batch.sh is for running simulations on supercomputer like Stanford Sherlock cluster. Please contact for more info

This repository contains three directories:
  1. src: all the program files
  2. data: where output from the simulation is stored
  3. zipped archives: where past versions and old files are stored
  
Goal of Simulation:
  1. To model how scientists choose to invest and research in ideas in real life
  2. To analyze factors that affect NIH research funding and how stressing those variables will affect output of scientists and their work
  3. To illustrate the power of connecting various discrete fields such as computer science, biomedicine, economics, and health policy

Any questions? Email lewis.zhang19@bcp.org

Special credits to Zach Templeton and Michelle Zhao as major contributors to the early stages of this project

## Helpful Images
How to run on terminal
![Image does not exist!](https://raw.githubusercontent.com/mzhao94/Scientist-Simulation/master/src/img/bash_run.png)

The overall project layout
![Image does not exist!](https://raw.githubusercontent.com/mzhao94/Scientist-Simulation/master/src/img/project_structure.png)

A preview of the server
![Image does not exist!](https://raw.githubusercontent.com/mzhao94/Scientist-Simulation/master/src/img/server_interactive.png)
