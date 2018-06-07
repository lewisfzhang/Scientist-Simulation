# run.py

from model import *
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from mesa.visualization.UserParam import UserSettableParameter
from numpy.random import poisson
import input
from mesa.batchrunner import BatchRunner
import matplotlib.pyplot as plt

use_server = True  # toggle between batch files and server (1 run)
use_slider = True  # only True when use_server is also True

# import variables from input
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

# can either use server to display interactive data (1 run), or do a batch of simultaneous runs
if not use_server:
    fixed_params = {"time_periods":time_periods, "ideas_per_time":ideas_per_time, "N":N, "true_sds_lam":true_sds_lam,
                    "true_means_lam":true_means_lam, "start_effort_lam":start_effort_lam, "start_effort_decay":start_effort_decay,
                    "k_lam":k_lam, "sds_lam":sds_lam, "means_lam":means_lam}
    variable_params = {"max_investment_lam": range(10,500,10)}
    model_reports = {"Total_Effort": get_total_effort}

    batch_run = BatchRunner(ScientistModel,
                            fixed_parameters=fixed_params,
                            variable_parameters=variable_params,
                            iterations=50,
                            max_steps=time_periods+2,
                            model_reporters=model_reports)
    batch_run.run_all()

    run_data = batch_run.get_model_vars_dataframe()
    run_data.head()
    # NOTE: obviously this is stupid since we only have 5 steps, but this is a template for future batch runs
    plt.scatter(run_data.max_investment_lam, run_data.Total_Effort)
    plt.show()

else:
    if use_slider:
        # sliders for ScientistModel(Model)
        time_periods = UserSettableParameter('slider', "Time Periods", 3, 1, 100, 1)
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

    chart1 = ChartModule([{"Label": "Total Effort",
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
