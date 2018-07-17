# Scientist-Simulation
Agent Based Model Simulation

Ways to run simulation:
  1. Via IDE
    - Create virtual environment, install necessary modules
    - Run src/run.py and then src/collect.py
  2. Via Bash (Preferred)
    - ./src/run.sh
    - Part 1: automatically checks to see if there is a working virtual environment, and install necessary files if not
    - Part 2: saves data as a tar.gz compressed file in data/zipped directory
    - Optional: you can specify 4 parameters
        - Time Periods (time_periods)
        - Ideas Per Time Period (ideas_per_time)
        - Number of Scientists Per Time Period (N)
        - Time Periods Alive (time_periods_alive)

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
