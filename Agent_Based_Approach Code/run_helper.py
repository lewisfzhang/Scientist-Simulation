import run
from model import *
import pandas as pd
import numpy as np
from run_graphs import *


# assigning which function to call in the run_graphs.py file
# FORMAT for other: (arg1, arg2, name1, name2, name3, extra_arg, file_name, linear)
def func_distr(type, *other):
    # data collecting
    model = run.model
    if type == "agent":
        #run.agent_df()
        print("hi")
    elif type == "model":
        # model dataframe
        model_vars = model.datacollector.get_model_vars_dataframe()
        # print("\n\n\nDATAFRAME (MODEL)\n",model_vars.to_string())
        model_vars.to_html('web/pages/page_model_vars.html')
        model_vars.to_csv('web/csv/csv_model_vars.csv')
    elif type == "data1":
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
        df_data1.to_html('web/pages/page_data1.html')
        df_data1.to_csv('web/csv/data1.csv')
    elif type == "ind_vars":
        ind_vars = {"agent_k_invested_ideas_flat": model.agent_k_invested_ideas_flat,
                    "agent_perceived_return_invested_ideas_flat": model.agent_perceived_return_invested_ideas_flat,
                    "agent_actual_return_invested_ideas_flat": model.agent_actual_return_invested_ideas_flat}
        df2 = pd.DataFrame.from_dict(ind_vars)
        df2.sort_values("agent_k_invested_ideas_flat", inplace=True)
        # print("\n\n\nDATAFRAME (IND VARS)\n", df2.to_string())
        df2.to_html('web/pages/page_ind_vars.html')
        df2.to_csv('web/csv/csv_ind_vars.csv')

    # graphs
    elif type == "im_graph":
        print("why")
        im_graph(*other)
    elif type == "resid_scatterplot":
        resid_scatterplot(*other)
    elif type == "two_var_scatterplot":
        two_var_scatterplot(*other)
    elif type == "two_var_bar_graph":
        two_var_bar_graph(*other)
    print("\nfinished task",type)
    stop_run()

#
# def multi_master():
#     p = run.p
#     # model = run.model
#     use_multiprocessing = run.use_multiprocessing
#
#     arg_list = [("agent", None), ("model", None), ("data1", None), ("ind_vars", None),
#
#                 ("im_graph", run.model.agent_k_invested_ideas_flat, run.model.agent_perceived_return_invested_ideas_flat, "k",
#                  "perceived returns",
#                  "perceived return vs cost for all INVESTED ideas across all scientists,time periods (biased)",
#                  False, "perceived", True),
#
#                 ("im_graph", run.model.agent_k_invested_ideas_flat, run.model.agent_perceived_return_invested_ideas_flat, "k",
#                  "perceived returns",
#                  "perceived return vs cost for all INVESTED ideas across all scientists,time periods (biased)",
#                  False, "perceived", False),
#
#                 ("im_graph", run.model.agent_k_invested_ideas_flat, run.model.agent_actual_return_invested_ideas_flat, "k",
#                  "actual returns",
#                  "actual return vs cost for all INVESTED ideas across all scientists,time periods (biased)", False,
#                  "actual", True),
#
#                 ("im_graph", run.model.agent_k_invested_ideas_flat, run.model.agent_actual_return_invested_ideas_flat, "k",
#                  "actual returns",
#                  "actual return vs cost for all INVESTED ideas across all scientists,time periods (biased)", False,
#                  "actual", False),
#
#                 ("resid_scatterplot", run.model.agent_actual_return_invested_ideas, run.model.agent_perceived_return_invested_ideas_flat,
#                 "Scientist ID", "Residual", "Residuals for all INVESTED ideas (actual-perceived)"),
#
#                 ("two_var_bar_graph", run.model.effort_invested_by_age, "Idea", "Marginal Effort Invested",
#                  "Marginal Effort Invested By Young and Old Scientists For All Ideas", True),
#
#                 ("two_var_bar_graph", run.model.effort_invested_by_age, "Idea", "Marginal Effort Invested",
#                  "Marginal Effort Invested By Young and Old Scientists For All Ideas", False),
#
#                 # runtime is WAY too long for linear y
#                 # ("two_var_scatterplot", model.avg_k, model.total_perceived_returns, "k", "perceived returns",
#                 #  "perceived return vs cost for INVESTED ideas (plot to check for bias)", True),
#
#                 ("two_var_scatterplot", run.model.avg_k, run.model.total_perceived_returns, "k", "perceived returns",
#                  "perceived return vs cost for INVESTED ideas (plot to check for bias)", False)]
#
#     if use_multiprocessing:
#         p.starmap(func_distr, arg_list)  # starmap maps each function call into a parallel thread
#     else:
#         for i in range(0, len(arg_list)):
#             func_distr(*arg_list[i])  # passes parameters in arg_list from list form to a series of arguments
#
#     # saves all of the images to an html file
#     png_to_html()