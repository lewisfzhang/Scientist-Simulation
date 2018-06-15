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
from multiprocessing import Pool

# start runtime
start = timeit.default_timer()

# switches that control how the program runs
use_server = False  # toggle between batch files and server (1 run)
use_slider = False  # only True when use_server is also True
use_batch = False
use_standard = True
draw_graphs = True
get_data = True
use_multiprocessing = True

# number of processors for multiprocessing
num_processors = input_file.num_processors

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
noise_factor = input_file.noise_factor
k_lam = input_file.k_lam
sds_lam = input_file.sds_lam
means_lam = input_file.means_lam
time_periods_alive = input_file.time_periods_alive

# default parameters for modelas a dictionary
all_params = {"time_periods": time_periods, "ideas_per_time": ideas_per_time, "N": N,
              "max_investment_lam": max_investment_lam, "true_sds_lam": true_sds_lam, "true_means_lam": true_means_lam,
              "start_effort_lam": start_effort_lam, "start_effort_decay": start_effort_decay,
              "noise_factor": noise_factor,
              "k_lam": k_lam, "sds_lam": sds_lam, "means_lam": means_lam, "time_periods_alive": time_periods_alive,
              "seed": seed}

# write parameters to text file
f = open('web/parameters.txt', 'w')
f.write(str(all_params))
f.close()

# set dataframe settings to max width, max rows, and max columns since we are collecting large quantities
# of data and printing out entire arrays/tuples
pd.set_option("display.max_colwidth", -1)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# print model and agent dataframes from simulation model run
def print_pd_data(model):
    # agent dataframe
    agent_vars = model.datacollector.get_agent_vars_dataframe()
    # print("\n\n\nDATAFRAME (AGENT)\n",agent_vars.to_string())
    # agent_vars.to_html('web/pages/page_agent_vars.html')  # page1
    agent_vars.to_csv('web/csv/csv_agent_vars.csv')

    # model dataframe
    model_vars = model.datacollector.get_model_vars_dataframe()
    # print("\n\n\nDATAFRAME (MODEL)\n",model_vars.to_string())
    # model_vars.to_html('web/pages/page_model_vars.html')  # page2
    model_vars.to_csv('web/csv/csv_model_vars.csv')

    # serious data table
    data1 = {"idea": range(0, model.total_ideas, 1),
             "TP": np.arange(model.total_ideas) // model.ideas_per_time,
             "scientists_invested": model.total_scientists_invested,
             "times_invested": model.total_times_invested,
             "avg_k": model.avg_k,
             "total_effort (marginal)": model.total_effort_tuple,
             "prop_invested": model.prop_invested,
             "total_pr": model.total_perceived_returns,
             "total_ar": model.total_actual_returns}
    df_data1 = pd.DataFrame.from_dict(data1)
    df_data1.to_html('web/pages/page_data1.html')  # page3
    model_vars.to_csv('web/csv/data1.csv')


# assigning which function to call in the run_graphs.py file
def func_distr(graph_type, arg1, arg2, name1, name2, name3, extra_arg, file_name, linear):
    start = timeit.default_timer()
    if graph_type == "im_graph":
        im_graph(arg1, arg2, name1, name2, name3, extra_arg, file_name, linear)
    elif graph_type == "resid_scatterplot":
        resid_scatterplot(arg1, arg2, name1, name2, name3)
    elif graph_type == "two_var_scatterplot":
        two_var_scatterplot(arg1, arg2, name1, name2, name3, linear)
    elif graph_type == "two_var_bar_graph":
        two_var_bar_graph(arg1, name1, name2, name3, linear)
    stop = timeit.default_timer()
    print(graph_type,stop-start,"seconds")


if __name__ == '__main__':  # for multiprocessor package so it knows the true main/run function
    # initiate multiprocessing with 'num_processors' threads
    # NOTE: increasing the number of processors does not always increase speed of program. in fact, it may actually
    # slow down the program due to the additional overhead needed for process switching
    p = Pool(num_processors)

    # printing parameters into console screen
    print("Variables:\n", all_params)

    # when we only want to run one model and collect all agent and model data from it
    if use_standard:
        print("compiled")
        stop_run(start)

        # initialize model object
        model = ScientistModel(time_periods, ideas_per_time, N, max_investment_lam, true_sds_lam, true_means_lam,
                               start_effort_lam, start_effort_decay, noise_factor, k_lam, sds_lam, means_lam, time_periods_alive, seed)

        for i in range(time_periods+2):
            model.step()
            print("step:",i)
            stop_run(start)

        if get_data:
            print_pd_data(model)
            print("finished accessing data tools")
            stop_run(start)

        if draw_graphs:
            # collect data from individual variables for plotting
            agent_k_invested_ideas = [a.final_k_invested_ideas for a in model.schedule.agents]
            agent_perceived_return_invested_ideas = [a.final_perceived_returns_invested_ideas for a in model.schedule.agents]
            agent_actual_return_invested_ideas = [a.final_actual_returns_invested_ideas for a in model.schedule.agents]

            # flattening numpy arrays
            agent_k_invested_ideas_flat = flatten_list(agent_k_invested_ideas)
            agent_perceived_return_invested_ideas_flat = flatten_list(agent_perceived_return_invested_ideas)
            agent_actual_return_invested_ideas_flat = flatten_list(agent_actual_return_invested_ideas)

            print("variable collecting finish")
            stop_run(start)

            ind_vars2 = {"agent_k_invested_ideas_flat": agent_k_invested_ideas_flat,
                         "agent_perceived_return_invested_ideas_flat": agent_perceived_return_invested_ideas_flat,
                         "agent_actual_return_invested_ideas_flat": agent_actual_return_invested_ideas_flat}
            df2 = pd.DataFrame.from_dict(ind_vars2)
            df2.sort_values("agent_k_invested_ideas_flat", inplace=True)
            # print("\n\n\nDATAFRAME 2 (IND VARS)\n", df2.to_string())
            # df2.to_html('web/pages/page_ind_vars2.html')  # page4
            df2.to_csv('web/csv/csv_ind_vars2.csv')

            print("ind var finish")
            stop_run(start)

            # reset_counter()

            arg_list = [("im_graph", agent_k_invested_ideas_flat, agent_perceived_return_invested_ideas_flat, "k", "perceived returns",
                        "perceived return vs cost for all INVESTED ideas across all scientists,time periods (biased)", False, "perceived", True),

                        ("im_graph", agent_k_invested_ideas_flat, agent_perceived_return_invested_ideas_flat, "k", "perceived returns",
                         "perceived return vs cost for all INVESTED ideas across all scientists,time periods (biased)", False, "perceived", False),

                        ("im_graph", agent_k_invested_ideas_flat, agent_actual_return_invested_ideas_flat, "k", "actual returns",
                        "actual return vs cost for all INVESTED ideas across all scientists,time periods (biased)", False, "actual", True),

                        ("im_graph", agent_k_invested_ideas_flat, agent_actual_return_invested_ideas_flat, "k", "actual returns",
                         "actual return vs cost for all INVESTED ideas across all scientists,time periods (biased)", False, "actual", False),

                        ("resid_scatterplot", agent_actual_return_invested_ideas, agent_perceived_return_invested_ideas_flat,
                        "TP", "Residual", "Residuals for all INVESTED ideas (actual-perceived)", None, None, None),

                        ("two_var_bar_graph", model.effort_invested_by_age, None, "Idea", "Marginal Effort Invested",
                        "Marginal Effort Invested By Young and Old Scientists For All Ideas", None, None, True),

                        ("two_var_bar_graph", model.effort_invested_by_age, None, "Idea", "Marginal Effort Invested",
                         "Marginal Effort Invested By Young and Old Scientists For All Ideas", None, None, False),

                        # runtime is WAY too long for linear y
                        # ("two_var_scatterplot", model.avg_k, model.total_perceived_returns, "k", "perceived returns",
                        #  "perceived return vs cost for INVESTED ideas (plot to check for bias)", None, None, True),

                        ("two_var_scatterplot", model.avg_k, model.total_perceived_returns, "k", "perceived returns",
                         "perceived return vs cost for INVESTED ideas (plot to check for bias)", None, None, False)]

            if use_multiprocessing:
                p.starmap(func_distr, arg_list)  # starmap maps each function call into a parallel thread
            else:
                for i in range(0, len(arg_list)):
                    func_distr(*arg_list[i])  # passes parameters in arg_list from list form to a series of arguments

            # plt.show()

            print("finished drawing graphs --> end of program")
            stop_run(start)

    # can either use server to display interactive data (1 run), or do a batch of simultaneous runs
    # use if we only care about model variables and how changing one variable affects the others
    # NOTE: cannot access agent variables
    if use_batch:
        fixed_params = {"time_periods":time_periods, "ideas_per_time":ideas_per_time, "N":N, "max_investment_lam":max_investment_lam,
                        "true_sds_lam":true_sds_lam, "true_means_lam":true_means_lam, "start_effort_lam":start_effort_lam,
                        "start_effort_decay":start_effort_decay, "noise_factor":noise_factor, "k_lam":k_lam, "sds_lam":sds_lam,
                        "means_lam":means_lam, "seed": randint(100000, 999999)}
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

    # sliders allow us to change certain variables in real time
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

    # launches an interactive display, probably don't need to implement this
    # NOTE: not practical if we are running large scale simulations as calculations will take too long to keep
    # up with the interactive display
    if use_server:
        chart1 = ChartModule([{"Label": "Total Effort",
                              "Color": "Black"}],
                               data_collector_name='datacollector')
        server = ModularServer(ScientistModel,
                               [chart1],
                               "Scientist Model",
                               all_params)

        server.port = 8521  # the default
        server.launch()
