# run.py

from model import ScientistModel
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from mesa.visualization.UserParam import UserSettableParameter
from numpy.random import poisson
import input

use_slider = True

if not use_slider:
    time_periods = input.time_periods
    ideas_per_time = input.ideas_per_time
    N = input.N
    max_investment_lam = input.max_investment_lam
    true_sds_lam = input.true_sds_lam
    true_means_lam = input.true_means_lam

    start_effort_lam = input.start_effort_lam
    start_effort_decay = input.start_effort_decay
    k_lam = input.k_lam
    sds_lam = input.sds_lam
    means_lam = input.means_lam

else:
    # sliders for ScientistModel(Model)
    time_periods = UserSettableParameter('slider', "Time Periods", 10, 1, 100, 1)
    ideas_per_time = UserSettableParameter('slider', "Ideas Per Time", 1, 1, 100, 1)
    N = UserSettableParameter('slider', "N", 2, 1, 100, 1)
    max_investment_lam = UserSettableParameter('slider', "Max Investment Lambda", 10, 1, 100, 1)
    true_sds_lam = UserSettableParameter('slider', "True SDS Lambda", 4, 1, 100, 1)
    true_means_lam = UserSettableParameter('slider', "True Means Lambda", 25, 1, 100, 1)

    # sliders for Scientist(Agent)
    start_effort_lam = UserSettableParameter('slider', "Start Effort Lambda", 10, 1, 100, 1)
    start_effort_decay = UserSettableParameter('slider', "Start Effort Decacy", 1, 1, 100, 1)
    k_lam = UserSettableParameter('slider', "K Lambda", 2, 1, 100, 1)
    sds_lam = UserSettableParameter('slider', "SDS Lambda", 4, 1, 100, 1)
    means_lam = UserSettableParameter('slider', "Means Lambda", 25, 1, 100, 1)

chart1 = ChartModule([{"Label": "Average Effort",
                      "Color": "Black"}],
                    data_collector_name='datacollector')
server = ModularServer(ScientistModel,
                       [chart1],
                       "Scientist Model",
                       {"time_periods":time_periods, "ideas_per_time":ideas_per_time, "N":N,
                        "max_investment_lam":max_investment_lam, "true_sds_lam":true_sds_lam,"true_means_lam":true_means_lam,
                        "start_effort_lam":start_effort_lam, "start_effort_decay":start_effort_decay, "k_lam":k_lam,
                        "sds_lam":sds_lam, "means_lam":means_lam})

server.port = 8521  # The default
server.launch()


# ORIGINAL CODE
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 14:05:55 2018

@author: zstemple
"""

# run.py

# import os
# os.chdir("C:/Users/zstemple/Documents/GitHub/Scientist-Simulation/Agent_Based_Approach Code")
# import sys
# orig_stdout = sys.stdout
# f = open("output.txt", "w")
# sys.stdout = f

# test_model = ScientistModel()
# NOTE: To iterate once through a model, the number of steps should be time
# periods + 2
# for i in range(5):
#     test_model.step()

# sys.stdout = orig_stdout
# f.close()
