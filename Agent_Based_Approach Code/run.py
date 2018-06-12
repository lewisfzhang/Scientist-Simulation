# run.py

from functions import *
from model import *
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from mesa.visualization.UserParam import UserSettableParameter
import input_file
from mesa.batchrunner import BatchRunner
import numpy as np
from run_graphs import *
import pandas as pd
import timeit
from random import randint


# runtime
start = timeit.default_timer()

use_server = False  # toggle between batch files and server (1 run)
use_slider = False  # only True when use_server is also True
use_batch = False
use_standard = True
draw_graphs = True
get_data = True

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
time_periods_alive = input_file.time_periods_alive

all_params = {"time_periods":time_periods, "ideas_per_time":ideas_per_time, "N":N,
            "max_investment_lam":max_investment_lam, "true_sds_lam":true_sds_lam,"true_means_lam":true_means_lam,
            "start_effort_lam":start_effort_lam, "start_effort_decay":start_effort_decay, "k_lam":k_lam,
            "sds_lam":sds_lam, "means_lam":means_lam, "time_periods_alive":time_periods_alive, "seed":seed}

pd.set_option("display.max_colwidth", -1)
pd.set_option('display.max_columns', None)

if use_standard:
    # initialize model object
    model = ScientistModel(time_periods, ideas_per_time, N, max_investment_lam, true_sds_lam, true_means_lam,
                           start_effort_lam, start_effort_decay, k_lam, sds_lam, means_lam, time_periods_alive, seed)
    for i in range(time_periods+2):
        model.step()

    print("\nVariables:\n",all_params)

    stop_run(start)

    if get_data:
        # agent dataframe
        agent_vars = model.datacollector.get_agent_vars_dataframe()
        # print("\n\n\nDATAFRAME (AGENT)\n",agent_vars.to_string())
        agent_vars.to_html('web/pages/page_agent_vars.html')  # page1
        agent_vars.to_csv('web/csv/csv_agent_vars.csv')

        # model dataframe
        model_vars = model.datacollector.get_model_vars_dataframe()
        # print("\n\n\nDATAFRAME (MODEL)\n",model_vars.to_string())
        model_vars.to_html('web/pages/page_model_vars.html')  # page2
        model_vars.to_csv('web/csv/csv_model_vars.csv')

        # serious data table
        avg_k = np.round(np.divide(model.total_k, model.total_scientists_invested,
                                   out=np.zeros_like(model.total_k), where=model.total_scientists_invested != 0), 2)
        # avg_pr = np.round(np.divide(model.total_perceived_returns, model.total_times_invested,
        #                    out=np.zeros_like(model.total_perceived_returns), where=model.total_times_invested!=0),2)
        # avg_ar = np.round(np.divide(model.total_actual_returns, model.total_times_invested,
        #                    out=np.zeros_like(model.total_actual_returns), where=model.total_times_invested != 0),2)
        prop_invested = np.round(np.divide(model.total_effort_tuple, model.max_investment,
                                 out=np.zeros_like(model.total_effort_tuple), where=model.max_investment != 0), 2)
        data1 = {"idea": range(0, model.total_ideas, 1),
                 "TP": np.arange(model.total_ideas) // model.ideas_per_time,
                 "scientists_invested": model.total_scientists_invested,
                 "times_invested": model.total_times_invested,
                 "avg_k": avg_k,
                 "total_effort (marginal)": model.total_effort_tuple,
                 "max_investment": model.max_investment,
                 "prop_invested": prop_invested,
                 "total_pr": np.round(model.total_perceived_returns, 2),
                 "total_ar": np.round(model.total_actual_returns, 2)}
        # "avg_pr": avg_pr,
        # "avg_ar": avg_ar}
        df_data1 = pd.DataFrame.from_dict(data1)
        df_data1.to_html('web/pages/page_data1.html')  # page3
        model_vars.to_csv('web/csv/data1.csv')

    stop_run(start)

    if draw_graphs:
        # collect data from individual variables for plotting
        agent_k_avail_ideas = [a.final_k_avail_ideas for a in model.schedule.agents]
        agent_perceived_return_avail_ideas = [a.final_perceived_returns_avail_ideas for a in model.schedule.agents]
        agent_actual_return_avail_ideas = [a.final_actual_returns_avail_ideas for a in model.schedule.agents]
        agent_k_invested_ideas = [a.final_k_invested_ideas for a in model.schedule.agents]
        agent_perceived_return_invested_ideas = [a.final_perceived_returns_invested_ideas for a in model.schedule.agents]
        agent_actual_return_invested_ideas = [a.final_actual_returns_invested_ideas for a in model.schedule.agents]

        agent_k_avail_ideas_flat = flatten_list(agent_k_avail_ideas)
        agent_perceived_return_avail_ideas_flat = flatten_list(agent_perceived_return_avail_ideas)
        agent_actual_return_avail_ideas_flat = flatten_list(agent_actual_return_avail_ideas)
        agent_k_invested_ideas_flat = flatten_list(agent_k_invested_ideas)
        agent_perceived_return_invested_ideas_flat = flatten_list(agent_perceived_return_invested_ideas)
        agent_actual_return_invested_ideas_flat = flatten_list(agent_actual_return_invested_ideas)

        # individual values dataframe 1
        ind_vars = {"agent_k_avail_ideas_flat":agent_k_avail_ideas_flat,
                    "agent_perceived_return_avail_ideas_flat":agent_perceived_return_avail_ideas_flat,
                    "agent_actual_return_avail_ideas_flat":agent_actual_return_avail_ideas_flat}
        df1 = pd.DataFrame.from_dict(ind_vars)
        df1.sort_values("agent_k_avail_ideas_flat", inplace=True)
        #print("\n\n\nDATAFRAME (IND VARS)\n", df1.to_string())
        df1.to_html('web/pages/page_ind_vars1.html')  # page4
        df1.to_csv('web/csv/csv_ind_vars1.html')

        ind_vars2 = {"agent_k_invested_ideas_flat":agent_k_invested_ideas_flat,
                    "agent_perceived_return_invested_ideas_flat":agent_perceived_return_invested_ideas_flat,
                    "agent_actual_return_invested_ideas_flat":agent_actual_return_invested_ideas_flat}
        df2 = pd.DataFrame.from_dict(ind_vars2)
        df2.sort_values("agent_k_invested_ideas_flat", inplace=True)
        # print("\n\n\nDATAFRAME 2 (IND VARS)\n", df2.to_string())
        df2.to_html('web/pages/page_ind_vars2.html')  # page5
        df2.to_csv('web/csv/csv_ind_vars2.html')

        reset_counter()

        # cost vs perceived return for all available ideas graph
        im_graph(agent_k_avail_ideas_flat, agent_perceived_return_avail_ideas_flat, "k", "perceived returns (1/100)",
                 "cost vs perceived return for all available ideas across all scientists,time periods (unbiased)", False)

        # cost vs actual return for all available ideas graph
        im_graph(agent_k_avail_ideas_flat, agent_actual_return_avail_ideas_flat, "k", "actual returns (1/100)",
                 "cost vs actual return for all available ideas across all scientists,time periods (unbiased)", False)

        # scatterplot of residuals for all available ideas graph
        # format: scatterplot(actual,perceived) | resid = actual-perceived
        # unflattened for time period calculation in scatterplot
        resid_scatterplot(agent_actual_return_avail_ideas,agent_perceived_return_avail_ideas_flat,
                    "TP","Residual","Residuals for all available ideas (actual-perceived, 1/100)")

        # cost vs perceived return for all INVESTED ideas graph
        im_graph(agent_k_invested_ideas_flat, agent_perceived_return_invested_ideas_flat, "k", "perceived returns (1/100)",
                 "cost vs perceived return for all INVESTED ideas across all scientists,time periods (biased)", False)

        # cost vs actual return for all INVESTED ideas graph
        im_graph(agent_k_invested_ideas_flat, agent_actual_return_invested_ideas_flat, "k", "actual returns (1/100)",
                 "cost vs actual return for all INVESTED ideas across all scientists,time periods (biased)", False)

        # scatterplot of residuals for all INVESTED ideas graph
        resid_scatterplot(agent_actual_return_invested_ideas, agent_perceived_return_invested_ideas_flat,
                    "TP", "Residual", "Residuals for all INVESTED ideas (actual-perceived, 1/100)")

        # Marginal Effort vs Idea (Young vs Old)
        two_var_bar_graph(model.effort_invested_by_age, "Idea", "Marginal Effort Invested",
                          "Marginal Effort Invested By Young and Old Scientists For All Ideas")

        # scatterplot of low/high K vs low/high PR for invested ideas
        mean_k = sum(agent_k_avail_ideas_flat)/len(agent_k_avail_ideas_flat)
        # divide sign (numerator and divisor too long), exclude 0's since they probably are never invested in model
        mean_pr = sum(agent_perceived_return_avail_ideas_flat) / \
                  (len(agent_perceived_return_avail_ideas_flat)-agent_perceived_return_avail_ideas_flat.count(0))
        two_var_scatterplot(avg_k, model.total_perceived_returns, "k", "perceived returns (1/100)",
                            "cost vs perceived return for INVESTED ideas (plot to check for bias)", mean_pr, mean_k)

        # plt.show()

    stop_run(start)

# can either use server to display interactive data (1 run), or do a batch of simultaneous runs
if use_batch:
    fixed_params = {"time_periods":time_periods, "ideas_per_time":ideas_per_time, "N":N, "max_investment_lam":max_investment_lam,
                    "true_sds_lam":true_sds_lam, "true_means_lam":true_means_lam, "start_effort_lam":start_effort_lam,
                    "start_effort_decay":start_effort_decay, "k_lam":k_lam, "sds_lam":sds_lam, "means_lam":means_lam,
                    "seed": randint(100000, 999999)}
    # NOTE: variable should not be the range, because you should only run 1 iteration of it
    variable_params = {"time_periods_alive":time_periods_alive}  # [min,max) total number of values in array is (max-min)/step
    model_reports = {"Total_Effort": get_total_effort}

    batch_run = BatchRunner(ScientistModel,
                            fixed_parameters=fixed_params,
                            variable_parameters=variable_params,
                            iterations=5,
                            max_steps=time_periods+2,
                            model_reporters=model_reports)
    batch_run.run_all()

    run_data = batch_run.get_model_vars_dataframe()
    # # NOTE: obviously this is stupid since we only have 5 steps, but this is a template for future batch runs
    # plt.scatter(run_data.max_investment_lam, run_data.Total_Effort)
    # plt.show()

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
