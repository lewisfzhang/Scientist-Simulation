# run.py

from functions import *
from model import *
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from mesa.visualization.UserParam import UserSettableParameter
from numpy.random import poisson
import input_file
from mesa.batchrunner import BatchRunner
import matplotlib.pyplot as plt
import numpy as np
from run_graphs import *
import math
import pandas as pd
import timeit

# runtime
start = timeit.default_timer()

use_server = False  # toggle between batch files and server (1 run)
use_slider = False  # only True when use_server is also True
use_batch = False
use_standard = True

# helper method for calculating runtime
def stop_run(start):
    # end runtime
    stop = timeit.default_timer()
    print("Runtime: ", stop - start, "seconds")

# import variables from input_file
seed = input_file.seed
time_periods = input_file.time_periods
ideas_per_time = input_file.ideas_per_time
N = input_file.N
max_investment_lam = input_file.max_investment_lam
true_sds_lam = input_file.true_sds_lam
true_means_lam = input_file.true_means_lam

start_effort_lam = input_file.start_effort_lam
start_effort_decay = input_file.start_effort_decay
k_lam = input_file.k_lam
sds_lam = input_file.sds_lam
means_lam = input_file.means_lam

all_params = {"time_periods":time_periods, "ideas_per_time":ideas_per_time, "N":N,
                            "max_investment_lam":max_investment_lam, "true_sds_lam":true_sds_lam,"true_means_lam":true_means_lam,
                            "start_effort_lam":start_effort_lam, "start_effort_decay":start_effort_decay, "k_lam":k_lam,
                            "sds_lam":sds_lam, "means_lam":means_lam, "seed":seed}

if use_standard:
    # initialize model object
    model = ScientistModel(time_periods, ideas_per_time, N, max_investment_lam, true_sds_lam, true_means_lam,
                           start_effort_lam, start_effort_decay, k_lam, sds_lam, means_lam, seed)
    for i in range(time_periods+2):
        model.step()

    # agent dataframe
    effort = model.datacollector.get_agent_vars_dataframe()
    # print("\n\nVariables\n",all_params,"\n\n\nDATAFRAME (AGENT)\n",effort.to_string())
    effort.to_html('web/pages/page'+str(page_counter())+'.html')

    # model dataframe
    ideas = model.datacollector.get_model_vars_dataframe()
    # ideas.to_html('web/test'+page_counter()+'.html')
    # print("\n\n\nDATAFRAME (MODEL)\n",ideas.to_string())
    ideas.to_html('web/pages/page'+str(page_counter())+'.html')

    # collect data from individual variables for plotting
    agent_k_avail_ideas = [a.final_k_avail_ideas for a in model.schedule.agents]
    agent_perceived_return_avail_ideas = [a.final_perceived_returns_avail_ideas for a in model.schedule.agents]
    agent_actual_return_avail_ideas = [a.final_actual_returns_avail_ideas for a in model.schedule.agents]

    agent_k_avail_ideas_flat = flatten_list(agent_k_avail_ideas)
    agent_perceived_return_avail_ideas_flat = flatten_list(agent_perceived_return_avail_ideas)
    agent_actual_return_avail_ideas_flat = flatten_list(agent_actual_return_avail_ideas)

    # individual values dataframe
    ind_vars = {"agent_k_avail_ideas_flat":agent_k_avail_ideas_flat,
                "agent_perceived_return_avail_ideas_flat":agent_perceived_return_avail_ideas_flat,
                "agent_actual_return_avail_ideas_flat":agent_actual_return_avail_ideas_flat}
    df1 = pd.DataFrame.from_dict(ind_vars)
    df1.sort_values("agent_k_avail_ideas_flat", inplace=True)
    # print("\n\n\nDATAFRAME (IND VARS)\n", df1.to_string())
    df1.to_html('web/pages/page'+str(page_counter())+'.html')

    reset_counter()

    # cost vs perceived return for all available ideas graph
    im_graph(agent_k_avail_ideas_flat, agent_perceived_return_avail_ideas_flat, "k", "perceived returns (1/1000)",
             "cost vs perceived return for all available ideas across all scientists,time periods (unbiased)", False)

    # cost vs actual return for all available ideas graph
    im_graph(agent_k_avail_ideas_flat, agent_actual_return_avail_ideas_flat, "k", "perceived returns (1/1000)",
             "cost vs actual return for all available ideas across all scientists,time periods (unbiased)", False)

    # scatterplot of residuals for all available ideas graph
    # format: scatterplot(actual,perceived) | resid = actual-perceived
    # unflattened for time period calculation in scatterplot
    scatterplot(agent_actual_return_avail_ideas,agent_perceived_return_avail_ideas_flat,
                "TP","Residual","Residuals for all available ideas (actual-perceived)")

    # cost vs perceived return for all INVESTED ideas graph

    # cost vs actual return for all INVESTED ideas graph

    # scatterplot of residuals for all INVESTED ideas graph

    # Marginal Effort vs Idea (Young vs Old)
    two_var_bar_graph(model.effort_invested_by_age, "Idea", "Marginal Effort Invested",
                      "Marginal Effort Invested By Young and Old Scientists For All Ideas")

    stop_run(start)

    # plt.show()



# can either use server to display interactive data (1 run), or do a batch of simultaneous runs
if use_batch:
    fixed_params = {"time_periods":time_periods, "ideas_per_time":ideas_per_time, "N":N, "true_sds_lam":true_sds_lam,
                    "true_means_lam":true_means_lam, "start_effort_lam":start_effort_lam, "start_effort_decay":start_effort_decay,
                    "k_lam":k_lam, "sds_lam":sds_lam, "means_lam":means_lam, "seed":seed}
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

if use_server:
    chart1 = ChartModule([{"Label": "Total Effort",
                          "Color": "Black"}],
                           data_collector_name='datacollector')
    server = ModularServer(ScientistModel,
                           [chart1],
                           "Scientist Model",
                           all_params)

    server.port = 8521  # The default
    server.launch()