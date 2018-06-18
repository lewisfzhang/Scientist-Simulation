# run.py

from functions import *
from model import *
import input_file
import numpy as np
from run_graphs import *
import pandas as pd
import timeit
from multiprocessing import *
import gc


def main():
    # start runtime
    start = timeit.default_timer()
    input_file.start = timeit.default_timer()

    # whether we want parallel processing (depending if it works on the system being run)
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
                  "max_investment_lam": max_investment_lam, "true_sds_lam": true_sds_lam,
                  "true_means_lam": true_means_lam,
                  "start_effort_lam": start_effort_lam, "start_effort_decay": start_effort_decay,
                  "noise_factor": noise_factor,
                  "k_lam": k_lam, "sds_lam": sds_lam, "means_lam": means_lam, "time_periods_alive": time_periods_alive,
                  "seed": seed}

    # set dataframe settings to max width, max rows, and max columns since we are collecting large quantities
    # of data and printing out entire arrays/tuples
    pd.set_option("display.max_colwidth", -1)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    print("\ncompiled/all variables stored")
    stop_run()

    # initialize model object
    model = ScientistModel(time_periods, ideas_per_time, N, max_investment_lam, true_sds_lam, true_means_lam,
                           start_effort_lam, start_effort_decay, noise_factor, k_lam, sds_lam, means_lam,
                           time_periods_alive, seed)

    print("\ninitialized model object")
    stop_run()

    # initiate multiprocessing with 'num_processors' threads
    # NOTE: increasing the number of processors does not always increase speed of program. in fact, it may actually
    # slow down the program due to the additional overhead needed for process switching
    # NOTE: fork doesn't work on Mac, spawn is best because it works on Mac and is default on Windows
    set_start_method('spawn')
    p = Pool(processes=num_processors)

    # printing parameters into console screen
    print("\nVariables:\n", all_params)

    # write parameters to text file
    f = open('web/parameters.txt', 'w')
    f.write(str(all_params))
    f.close()

    print("\nentering main function")
    stop_run()

    for i in range(time_periods+2):
        model.step()
        print("\nstep:",i)
        stop_run()

    print("\nTOTAL TIME TO FINISH RUNNING SIMULATION:", timeit.default_timer() - start, "seconds")

    agent_var = model.datacollector.get_agent_vars_dataframe()
    model_var = model.datacollector.get_model_vars_dataframe()

    data1_dict = {"idea": range(0, model.total_ideas, 1),
                 "TP": np.arange(model.total_ideas) // model.ideas_per_time,
                 "scientists_invested": model.total_scientists_invested,
                 "times_invested": model.total_times_invested,
                 "avg_k": model.avg_k,
                 "total_effort (marginal)": model.total_effort_tuple,
                 "prop_invested": model.prop_invested,
                 "total_pr": model.total_perceived_returns,
                 "total_ar": model.total_actual_returns}

    ind_vars_dict = {"agent_k_invested_ideas_flat": model.agent_k_invested_ideas_flat,
                     "agent_perceived_return_invested_ideas_flat": model.agent_perceived_return_invested_ideas_flat,
                     "agent_actual_return_invested_ideas_flat": model.agent_actual_return_invested_ideas_flat}

    arg_list = [("agent", agent_var), ("model", model_var), ("data1", data1_dict), ("ind_vars", ind_vars_dict),

                ("im_graph", model.agent_k_invested_ideas_flat, model.agent_perceived_return_invested_ideas_flat, "k", "perceived returns",
                "perceived return vs cost for all INVESTED ideas across all scientists,time periods (biased)", False, "perceived", True),

                ("im_graph", model.agent_k_invested_ideas_flat, model.agent_perceived_return_invested_ideas_flat, "k", "perceived returns",
                 "perceived return vs cost for all INVESTED ideas across all scientists,time periods (biased)", False, "perceived", False),

                ("im_graph", model.agent_k_invested_ideas_flat, model.agent_actual_return_invested_ideas_flat, "k", "actual returns",
                "actual return vs cost for all INVESTED ideas across all scientists,time periods (biased)", False, "actual", True),

                ("im_graph", model.agent_k_invested_ideas_flat, model.agent_actual_return_invested_ideas_flat, "k", "actual returns",
                 "actual return vs cost for all INVESTED ideas across all scientists,time periods (biased)", False, "actual", False),

                ("resid_scatterplot", model.agent_actual_return_invested_ideas, model.agent_perceived_return_invested_ideas_flat,
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

    if use_multiprocessing:
        p.starmap(func_distr, arg_list)  # starmap maps each function call into a parallel thread
    else:
        for i in range(0, len(arg_list)):
            func_distr(*arg_list[i])  # passes parameters in arg_list from list form to a series of arguments

    # saves all of the images to an html file
    png_to_html()

    print("\nTotal time to process data")
    stop_run()

    print("\nEND OF PROGRAM\ntotal runtime:", timeit.default_timer() - start, "seconds")


# assigning which function to call in the run_graphs.py file
def func_distr(graph_type, *other):
    start = timeit.default_timer()

    if graph_type == "agent":
        # agent dataframe (other[0] contains agent_vars)
        agent_vars = other[0]
        # print("\n\n\nDATAFRAME (AGENT)\n",agent_vars.to_string())
        agent_vars.to_html('web/pages/page_agent_vars.html')  # page1
        agent_vars.to_csv('web/csv/csv_agent_vars.csv')
    elif graph_type == "model":
        # model dataframe (other[0] contains model_vars)
        model_vars = other[0]
        # print("\n\n\nDATAFRAME (MODEL)\n",model_vars.to_string())
        # model_vars.to_html('web/pages/page_model_vars.html')  # page2
        model_vars.to_csv('web/csv/csv_model_vars.csv')
    elif graph_type == "data1":
        # serious data table
        data1 = other[0]
        df_data1 = pd.DataFrame.from_dict(data1)
        df_data1.to_html('web/pages/page_data1.html')  # page3
        df_data1.to_csv('web/csv/data1.csv')
    elif graph_type == "ind_vars":
        ind_vars = other[0]
        df_ind_vars = pd.DataFrame.from_dict(ind_vars)
        df_ind_vars.sort_values("agent_k_invested_ideas_flat", inplace=True)
        # print("\n\n\nDATAFRAME (IND VARS)\n", df_ind_vars.to_string())
        # df_ind_vars.to_html('web/pages/page_ind_vars.html')
        df_ind_vars.to_csv('web/csv/csv_ind_vars.csv')
    elif graph_type == "im_graph":
        im_graph(*other)
    elif graph_type == "resid_scatterplot":
        resid_scatterplot(*other)
    elif graph_type == "two_var_scatterplot":
        two_var_scatterplot(*other)
    elif graph_type == "two_var_bar_graph":
        two_var_bar_graph(*other)

    stop = timeit.default_timer()
    print("\nfinished", graph_type, stop-start, "seconds")
    collected = gc.collect()
    print("Collected",collected,"objects")


if __name__ == '__main__':  # for multiprocessor package so it knows the true main/run function
    main()