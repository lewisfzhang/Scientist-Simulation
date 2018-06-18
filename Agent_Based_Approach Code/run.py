# run.py

from functions import *
from model import *
import input_file
import pandas as pd
import timeit
from multiprocessing import *
import run_helper
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from mesa.visualization.UserParam import UserSettableParameter
from mesa.batchrunner import BatchRunner

# start runtime
start = timeit.default_timer()
input_file.start = timeit.default_timer()

# switches that control how the program runs
# use_server = False  # toggle between batch files and server (1 run)
# use_slider = False  # only True when use_server is also True
# use_batch = False
use_standard = True
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

# set dataframe settings to max width, max rows, and max columns since we are collecting large quantities
# of data and printing out entire arrays/tuples
pd.set_option("display.max_colwidth", -1)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# initiate multiprocessing with 'num_processors' threads
# NOTE: increasing the number of processors does not always increase speed of program. in fact, it may actually
# slow down the program due to the additional overhead needed for process switching
p = Pool(num_processors)

# printing parameters into console screen
print("\nVariables:\n", all_params)

# write parameters to text file
f = open('web/parameters.txt', 'w')
f.write(str(all_params))
f.close()

print("\ncompiled")
stop_run()

# initialize model object
model = ScientistModel(time_periods, ideas_per_time, N, max_investment_lam, true_sds_lam, true_means_lam,
                       start_effort_lam, start_effort_decay, noise_factor, k_lam, sds_lam, means_lam, time_periods_alive, seed)

print("\ninitialized model object")
stop_run()


def agent_df():
    global model
    # agent dataframe
    agent_vars = model.datacollector.get_agent_vars_dataframe()
    # print("\n\n\nDATAFRAME (AGENT)\n",agent_vars.to_string())
    agent_vars.to_html('web/pages/page_agent_vars.html')
    agent_vars.to_csv('web/csv/csv_agent_vars.csv')
    print("done")


def main():
    global p
    global model
    global use_multiprocessing

    # when we only want to run one model and collect all agent and model data from it
    if use_standard:

        for i in range(time_periods+2):
            model.step()
            print("\nstep:",i)
            stop_run()

        agent_df()

        arg_list = [("agent", None), ("model", None), ("data1", None), ("ind_vars", None),

                    ("im_graph", model.agent_k_invested_ideas_flat,
                     model.agent_perceived_return_invested_ideas_flat, "k",
                     "perceived returns",
                     "perceived return vs cost for all INVESTED ideas across all scientists,time periods (biased)",
                     False, "perceived", True),

                    ("im_graph", model.agent_k_invested_ideas_flat,
                     model.agent_perceived_return_invested_ideas_flat, "k",
                     "perceived returns",
                     "perceived return vs cost for all INVESTED ideas across all scientists,time periods (biased)",
                     False, "perceived", False),

                    ("im_graph", model.agent_k_invested_ideas_flat,
                     model.agent_actual_return_invested_ideas_flat, "k",
                     "actual returns",
                     "actual return vs cost for all INVESTED ideas across all scientists,time periods (biased)", False,
                     "actual", True),

                    ("im_graph", model.agent_k_invested_ideas_flat,
                     model.agent_actual_return_invested_ideas_flat, "k",
                     "actual returns",
                     "actual return vs cost for all INVESTED ideas across all scientists,time periods (biased)", False,
                     "actual", False),

                    ("resid_scatterplot", model.agent_actual_return_invested_ideas,
                     model.agent_perceived_return_invested_ideas_flat,
                     "Scientist ID", "Residual", "Residuals for all INVESTED ideas (actual-perceived)"),

                    ("two_var_bar_graph", model.effort_invested_by_age, "Idea", "Marginal Effort Invested",
                     "Marginal Effort Invested By Young and Old Scientists For All Ideas", True),

                    ("two_var_bar_graph", model.effort_invested_by_age, "Idea", "Marginal Effort Invested",
                     "Marginal Effort Invested By Young and Old Scientists For All Ideas", False),

                    # runtime is WAY too long for linear y
                    # ("two_var_scatterplot", model.avg_k, model.total_perceived_returns, "k", "perceived returns",
                    #  "perceived return vs cost for INVESTED ideas (plot to check for bias)", True),

                    ("two_var_scatterplot", model.avg_k, model.total_perceived_returns, "k", "perceived returns",
                    "perceived return vs cost for INVESTED ideas (plot to check for bias)", False)]

        p_workers = []
        if use_multiprocessing:
            # p_workers.append(Process(target=run_helper.func_distr, args=arg_list[4]))
            # p_workers[0].start()
            # p_workers[0].join()
            # for i in range(len(arg_list)):
            #     p_object = Process(target=run_helper.func_distr, args=arg_list[i])
            #     p_object.daemon = True
            #     p_workers.append(p_object)
            # for i in range(len(arg_list)):
            #     p_workers[i].start()
            # for i in range(len(arg_list)):
            #     p_workers[i].join()
            p.starmap(run_helper.func_distr, arg_list)  # starmap maps each function call into a parallel thread
        else:
            for i in range(0, len(arg_list)):
                run_helper.func_distr(*arg_list[i])  # passes parameters in arg_list from list form to a series of arguments

        # saves all of the images to an html file
        png_to_html()

        print("\nend of program", "\ntotal runtime:", timeit.default_timer() - start)


if __name__ == '__main__':  # for multiprocessor package so it knows the true main/run function
    main()

# # initiate multiprocessing with 'num_processors' threads
# # NOTE: increasing the number of processors does not always increase speed of program. in fact, it may actually
# # slow down the program due to the additional overhead needed for process switching
# p = Pool(num_processors)
#
# # printing parameters into console screen
# print("Variables:\n", all_params)
#
# # write parameters to text file
# f = open('web/parameters.txt', 'w')
# f.write(str(all_params))
# f.close()
#
# # when we only want to run one model and collect all agent and model data from it
# if use_standard:
#     print("\ncompiled")
#     stop_run()
#
#     # initialize model object
#     model = ScientistModel(time_periods, ideas_per_time, N, max_investment_lam, true_sds_lam, true_means_lam,
#                            start_effort_lam, start_effort_decay, noise_factor, k_lam, sds_lam, means_lam, time_periods_alive, seed)
#
#     for i in range(time_periods+2):
#         model.step()
#         print("\nstep:",i)
#         stop_run()
#
#     command_list = ["data", "graph"]
#     if use_multiprocessing:
#         p.map(multi_master, command_list)
#     else:
#         for i in range(len(command_list)):
#             multi_master(command_list[i])
#
#     print("\nend of program", "\ntotal runtime:", timeit.default_timer() - start)
#
# # # can either use server to display interactive data (1 run), or do a batch of simultaneous runs
# # # use if we only care about model variables and how changing one variable affects the others
# # # NOTE: cannot access agent variables
# # if use_batch:
# #     fixed_params = {"time_periods":time_periods, "ideas_per_time":ideas_per_time, "N":N, "max_investment_lam":max_investment_lam,
# #                     "true_sds_lam":true_sds_lam, "true_means_lam":true_means_lam, "start_effort_lam":start_effort_lam,
# #                     "start_effort_decay":start_effort_decay, "noise_factor":noise_factor, "k_lam":k_lam, "sds_lam":sds_lam,
# #                     "means_lam":means_lam, "seed": randint(100000, 999999)}
# #     # NOTE: variable should not be the range, because you should only run 1 iteration of it
# #     variable_params = {"time_periods_alive":time_periods_alive}  # [min,max) total number of values in array is (max-min)/step
# #     model_reports = {"Total_Effort": get_total_effort}
# #
# #     batch_run = BatchRunner(ScientistModel,
# #                             fixed_parameters=fixed_params,
# #                             variable_parameters=variable_params,
# #                             iterations=5,
# #                             max_steps=time_periods+2,
# #                             model_reporters=model_reports)
# #     batch_run.run_all()
# #     run_data = batch_run.get_model_vars_dataframe()
# #     # # NOTE: obviously this is stupid since we only have 5 steps, but this is a template for future batch runs
# #     # plt.scatter(run_data.max_investment_lam, run_data.Total_Effort)
# #     # plt.show()
# #
# # # sliders allow us to change certain variables in real time
# # if use_slider:
# #     # sliders for ScientistModel(Model)
# #     time_periods = UserSettableParameter('slider', "Time Periods", 3, 1, 100, 1)
# #     ideas_per_time = UserSettableParameter('slider', "Ideas Per Time", 1, 1, 100, 1)
# #     N = UserSettableParameter('slider', "N", 2, 1, 100, 1)
# #     max_investment_lam = UserSettableParameter('slider', "Max Investment Lambda", 10, 1, 100, 1)
# #     true_sds_lam = UserSettableParameter('slider', "True SDS Lambda", 4, 1, 100, 1)
# #     true_means_lam = UserSettableParameter('slider', "True Means Lambda", 25, 1, 100, 1)
# #
# #     # sliders for Scientist(Agent)
# #     start_effort_lam = UserSettableParameter('slider', "Start Effort Lambda", 10, 1, 100, 1)
# #     start_effort_decay = UserSettableParameter('slider', "Start Effort Decacy", 1, 1, 100, 1)
# #     k_lam = UserSettableParameter('slider', "K Lambda", 2, 1, 100, 1)
# #     sds_lam = UserSettableParameter('slider', "SDS Lambda", 4, 1, 100, 1)
# #     means_lam = UserSettableParameter('slider', "Means Lambda", 25, 1, 100, 1)
# #
# # # launches an interactive display, probably don't need to implement this
# # # NOTE: not practical if we are running large scale simulations as calculations will take too long to keep
# # # up with the interactive display
# # if use_server:
# #     chart1 = ChartModule([{"Label": "Total Effort",
# #                           "Color": "Black"}],
# #                            data_collector_name='datacollector')
# #     server = ModularServer(ScientistModel,
# #                            [chart1],
# #                            "Scientist Model",
# #                            all_params)
# #
# #     server.port = 8521  # the default
# #     server.launch()
